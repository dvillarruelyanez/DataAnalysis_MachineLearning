import os
import glob
import random
import copy
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import tifffile as tiff
import cv2
from scipy.ndimage import shift as scipy_shift
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift
from tqdm import tqdm

class AugmentedSequenceDataset(Dataset):
    """
    Wraps a base sequence dataset to apply spatial augmentations consistently across time.

    Args:
        tensor_dataset (Dataset): The underlying dataset returning (X_seq, y_seq).
    """
    def __init__(self, tensor_dataset: Dataset):
        self.tensor_dataset = tensor_dataset
        
        # Spatial augmentation pipeline.
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.RandomRotation(180)],
                p=0.5
            )
        ])

    def __len__(self) -> int:
        return len(self.tensor_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # X_seq: (T_in, 1, H, W)
        # y_seq: (T_out, 1, H, W)
        X_seq, y_seq = self.tensor_dataset[idx]

        T_in = X_seq.shape[0]
        T_out = y_seq.shape[0]

        # Merge sequences to treat Time as Channels
        # We apply the same transform to all frames.
        
        X_sq = X_seq.squeeze(1) # (T_in, H, W)
        y_sq = y_seq.squeeze(1) # (T_out, H, W)

        # Combined shape: (T_in + T_out, H, W)
        combined = torch.cat([X_sq, y_sq], dim=0)

        #  Apply transformations
        combined_aug = self.transform(combined)

        #  Split back into input and target sequences
        X_aug = combined_aug[:T_in]            # (T_in, H, W)
        y_aug = combined_aug[T_in:T_in+T_out]  # (T_out, H, W)

        # Restore Channel Dimension -> (T, 1, H, W)
        X_aug = X_aug.unsqueeze(1)
        y_aug = y_aug.unsqueeze(1)

        return X_aug, y_aug


class EmbryoDataset(Dataset):
    """
    A custom Dataset that references pre-loaded data stacks and creates sliding windows on-the-fly.
    
    """
    def __init__(self, data_source: Dict[str, np.ndarray], valid_indices: List[Tuple[str, int]], n_past: int, n_future: int):
        """
        Args:
            data_source (dict): Dictionary mapping embryo_ids to full 3D numpy stacks (T, H, W).
            valid_indices (list): List of tuples (embryo_key, start_index) defining valid sequences.
            n_past (int): Number of input frames.
            n_future (int): Number of target frames.
        """
        self.data_source = data_source
        self.valid_indices = valid_indices
        self.n_past = n_past
        self.n_future = n_future

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Retrieve the pointer to the sequence
        embryo_key, start_idx = self.valid_indices[idx]
        
        # Get reference to the full stack
        full_stack = self.data_source[embryo_key] # Shape: (Total_Frames, H, W)
        
        # Calculate slicing coordinates
        past_end = start_idx + self.n_past
        future_end = past_end + self.n_future
        
        # Slice the arrays (Creates a view or small copy)
        X_seq_np = full_stack[start_idx : past_end]  # (T_past, H, W)
        y_seq_np = full_stack[past_end : future_end] # (T_fut, H, W)
        
        # Add Channel Dimension (Grayscale) -> (T, 1, H, W)
        X_seq_np = np.expand_dims(X_seq_np, axis=1) 
        y_seq_np = np.expand_dims(y_seq_np, axis=1)

        # Convert to Tensor
        X_t = torch.from_numpy(X_seq_np).float()
        y_t = torch.from_numpy(y_seq_np).float()
        
        return X_t, y_t


class DataModule:
    """
    Manages the entire data pipeline: Loading, Preprocessing, Normalization, and Batching.
    
    Pipeline:
        1. Load raw TIFF stacks (Images and Masks).
        2. Drift Correction and Auto-Crop.
        3. Resize & Normalize (Global percentile-based normalization).
        4. Split into Train/Val/Test (By Embryo ID).
    """
    def __init__(self, data_dir: str, target_size: Tuple[int, int] = (256, 256), t_start: int = 120, t_end: int = 960):
        self.data_dir = data_dir
        self.target_size = target_size
        self.t_start = t_start
        self.t_end = t_end
        
        # Processed data storage
        self.normalized_data = {}      
        self.metadata_timestamps = {}
        self.raw_paths = {}
        
        self.train_keys = []
        self.test_keys = []
        
        # Placeholders for loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _calculate_timestamps(self, num_frames: int) -> np.ndarray:
        return np.linspace(self.t_start, self.t_end, num_frames)

    def _correct_drift(self, img_stack: np.ndarray, mask_stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Corrects mechanical drift using sequential alignment (t vs t-1).
        
        """
        corrected_imgs = []
        corrected_masks = []
        
        # Initialize containers with the first frame
        corrected_imgs.append(img_stack[0])
        corrected_masks.append(mask_stack[0])
        
        # Cumulative shift (dy, dx). Starts at (0,0)
        current_drift = np.array([0.0, 0.0])
        
        # We iterate from the second frame onwards
        for i in range(1, len(img_stack)):
            prev_img = img_stack[i-1] # Reference is the PREVIOUS frame
            curr_img = img_stack[i]
            
            # Calculate shift between current and previous frame
            rel_shift, _, _ = phase_cross_correlation(
                prev_img, curr_img,
                upsample_factor=5,
                normalization=None 
            )
            
            # Update global drift accumulator
            current_drift += rel_shift
            
            # Apply the cumulative shift to the current raw frame
            shifted_img = scipy_shift(curr_img, shift=current_drift, mode='constant', cval=0)
            shifted_mask = scipy_shift(mask_stack[i], shift=current_drift, mode='constant', cval=0)
            
            corrected_imgs.append(shifted_img)
            corrected_masks.append(shifted_mask)
            
        return np.array(corrected_imgs), np.array(corrected_masks)

    def _auto_crop(self, img_stack: np.ndarray, mask_stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Determines the bounding box of the embryo's movement over the entire timeline.
        It uses a Maximum Intensity Projection (MIP) of the masks to find the union of all occupied areas.
        """
        # Temporal Projection: Collapses time to see everywhere the embryo has been
        projection = np.max(mask_stack, axis=0).astype(np.uint8) 
        projection[projection > 0] = 1
            
        coords = cv2.findNonZero(projection)

        if coords is None:
            # Fallback for empty images
            return img_stack, mask_stack
    
        # Calculate Bounding Box
        x, y, w, h = cv2.boundingRect(coords)
        
        # Add padding (5%)
        max_side = max(w, h)
        padding = int(max_side * 0.05)
        new_side = max_side + padding
        
        # Center the crop
        H_orig, W_orig = img_stack.shape[1], img_stack.shape[2]
        center_x = x + w // 2
        center_y = y + h // 2
        
        start_x = max(0, center_x - new_side // 2)
        start_y = max(0, center_y - new_side // 2)
        end_x = start_x + new_side
        end_y = start_y + new_side
        
        # Crop logic with canvas (handles out-of-bounds by padding with zeros)
        img_stack_cropped = []
        mask_stack_cropped = []
        
        for i in range(len(img_stack)):
            canvas_img = np.zeros((new_side, new_side), dtype=np.float32)
            canvas_mask = np.zeros((new_side, new_side), dtype=np.float32)
            
            src_x1 = max(0, start_x); src_y1 = max(0, start_y)
            src_x2 = min(W_orig, end_x); src_y2 = min(H_orig, end_y)
            
            dst_x1 = max(0, src_x1 - start_x); dst_y1 = max(0, src_y1 - start_y)
            dst_x2 = dst_x1 + (src_x2 - src_x1); dst_y2 = dst_y1 + (src_y2 - src_y1)
            
            if (dst_x2 > dst_x1) and (dst_y2 > dst_y1):
                canvas_img[dst_y1:dst_y2, dst_x1:dst_x2] = img_stack[i][src_y1:src_y2, src_x1:src_x2]
                canvas_mask[dst_y1:dst_y2, dst_x1:dst_x2] = mask_stack[i][src_y1:src_y2, src_x1:src_x2]
            
            img_stack_cropped.append(canvas_img)
            mask_stack_cropped.append(canvas_mask)
            
        return np.array(img_stack_cropped), np.array(mask_stack_cropped)

    def _process_stack(self, img_path: str, mask_path: str) -> np.ndarray:
        """
        Loads, corrects drift, crops, masks, resizes, and normalizes a single embryo stack.
        """
        img_stack = tiff.imread(img_path)
        mask_stack = tiff.imread(mask_path)
        
        # Sync lengths
        min_len = min(len(img_stack), len(mask_stack))
        img_stack = img_stack[:min_len]
        mask_stack = mask_stack[:min_len]

        img_stack, mask_stack = self._correct_drift(img_stack, mask_stack)
        
        try:
            img_stack, mask_stack = self._auto_crop(img_stack, mask_stack)
        except Exception as e:
            print(f"Auto crop warning for {os.path.basename(img_path)}: {e}")
        
        # Binarize mask
        mask_stack = (mask_stack > 0).astype(np.float32)
        
        T = len(img_stack)
        target_wh = (self.target_size[1], self.target_size[0])
    
        resized_masked_imgs = []
        
        # Masking & Resizing
        for i in range(T):
            img = img_stack[i].astype(np.float32)
            mask = mask_stack[i]
            
            img_masked = img * mask  
            
            img_resized = cv2.resize(
                img_masked,
                target_wh,
                interpolation=cv2.INTER_LINEAR
            )
            resized_masked_imgs.append(img_resized)
    
        resized_masked_imgs = np.stack(resized_masked_imgs, axis=0) # (T, H, W)
        
        # Global Normalization
        # We calculate percentiles over the FULL temporal stack to avoid flickering artifacts.
        full_stack = resized_masked_imgs.reshape(-1)
        p2_global, p99_global = np.percentile(full_stack, (0.0, 100.0))
        
        if p99_global <= p2_global:
            p99_global = p2_global + 1e-6
        
        # Normalize to [0, 1]
        stack_norm = (resized_masked_imgs - p2_global) / (p99_global - p2_global)
        stack_norm = np.clip(stack_norm, 0.0, 1.0)
        
        return stack_norm.astype(np.float32)

    def load_and_process_data(self):
        """
        Iterates through train/test folders and populates the internal data dictionary.
        """
        print(f"Scanning directory: {self.data_dir} ...")
        
        for split in ['train', 'test']:
            images_dir = os.path.join(self.data_dir, split, 'images')
            masks_dir = os.path.join(self.data_dir, split, 'masks')
            
            if not os.path.exists(images_dir):
                print(f"Warning: {images_dir} not found.")
                continue
                
            image_files = glob.glob(os.path.join(images_dir, "*o.tif"))
            print(f"Found {len(image_files)} images in {split} set.")
            
            for img_path in tqdm(image_files, desc=f"Processing {split}"):
                filename = os.path.basename(img_path)
                mask_filename = filename.replace('o.tif', '.tif')
                mask_path = os.path.join(masks_dir, mask_filename)
                
                if not os.path.exists(mask_path):
                    continue
                
                try:
                    final_stack = self._process_stack(img_path, mask_path)
                    
                    # Key generation: 'E1_o.tif' -> 'E1_'
                    key_name = filename.split('.')[0].replace('o', '') 
                    
                    self.normalized_data[key_name] = final_stack
                    self.metadata_timestamps[key_name] = self._calculate_timestamps(len(final_stack))
                    self.raw_paths[key_name] = img_path
                    
                    if split == 'train':
                        self.train_keys.append(key_name)
                    else:
                        self.test_keys.append(key_name)
                        
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

        print(f"Loaded {len(self.train_keys)} training stacks.")
        print(f"Loaded {len(self.test_keys)} test stacks.")

    def _create_sequences(self, data_stack: np.ndarray, n_past: int, n_future: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Helper to create sliding windows for testing/visualization.
        """
        X_list, y_list = [], []
        total_frames = len(data_stack)

        for i in range(total_frames - n_past - n_future + 1):
            past_start = i
            past_end = past_start + n_past
            future_start = past_end
            future_end = future_start + n_future

            X_seq = data_stack[past_start : past_end]
            y_seq = data_stack[future_start : future_end]
            X_list.append(X_seq)
            y_list.append(y_seq)

        return np.array(X_list), np.array(y_list)

    def prepare_dataloaders(self, n_past: int, n_future: int, train_split_percent: float, batch_size: int, use_augmentation: bool = False):
        """
        Prepares the DataLoaders for training, validation, and testing.
        
        This method executes a grouped split based on Embryo ID.

        Args:
            n_past (int): Number of past (context) frames input to the model.
            n_future (int): Number of future frames to predict (prediction horizon).
            train_split_percent (float): Proportion of embryos allocated to the training set.
            batch_size (int): Number of sequences per batch.
            use_augmentation (bool): If True, applies consistent spatial transformations 
                                     (flips, rotations) to the training sequences.
        """
        self.n_past = n_past
        self.n_future = n_future
        
        # 1. Split Strategy: Grouped by Embryo ID
        all_train_keys = self.train_keys.copy()
        random.seed(42)  # Fixed seed for reproducibility
        random.shuffle(all_train_keys)
        
        n_split = int(len(all_train_keys) * train_split_percent)
        actual_train_names = all_train_keys[:n_split]
        actual_val_names = all_train_keys[n_split:]
        
        print(f"Splitting Logic (By Embryo ID): Train={len(actual_train_names)}, Val={len(actual_val_names)}")

        # 2. Temporal Subsampling (Stride) Configuration
        TRAIN_STRIDE = 1
        VAL_STRIDE = 1

        # Helper function to generate valid (Embryo_Key, Start_Index) tuples
        def get_indices(names_list: List[str], stride: int = 1) -> List[Tuple[str, int]]:
            indices = []
            for name in names_list:
                stack = self.normalized_data[name]
                # Calculate total valid starting positions for a sequence of length (n_past + n_future)
                num_sequences = len(stack) - n_past - n_future + 1
                
                if num_sequences > 0:
                    # Apply stride step to the range generator
                    indices.extend([(name, i) for i in range(0, num_sequences, stride)])
            return indices

        # Generate indices with defined strides
        train_indices = get_indices(actual_train_names, stride=TRAIN_STRIDE)
        val_indices = get_indices(actual_val_names, stride=VAL_STRIDE)
        
        # For Testing, we typically use Stride=1 (Dense Evaluation) or match validation
        # to ensure we capture metrics across the entire fine-grained timeline.
        test_indices = get_indices(self.test_keys, stride=1)
        
        print(f"Total Sequences Generated: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

        # 3. Dataset Instantiation
        self.train_dataset = EmbryoDataset(self.normalized_data, train_indices, n_past, n_future)
        self.val_dataset = EmbryoDataset(self.normalized_data, val_indices, n_past, n_future)
        self.test_dataset = EmbryoDataset(self.normalized_data, test_indices, n_past, n_future)

        # 4. Augmentation Wrapper
        if use_augmentation:
            # Apply spatial augmentations (Rotation/Flips) consistently across time
            train_ds = AugmentedSequenceDataset(self.train_dataset)
            print("Augmentation pipeline enabled for training set.")
        else:
            train_ds = self.train_dataset

        # 5. Loader Creation
        # num_workers=0 is used to avoid IPC overhead in simple setups, 
        # but can be increased if CPU becomes the bottleneck.
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        
        # Store references to the split names for logging/visualization later
        self.train_val_names = self.train_keys
        self.test_names = self.test_keys
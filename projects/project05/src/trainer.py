import copy
import time
import random
from typing import Optional, Callable, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

class Trainer:
    """
    Handles the training, validation, and testing lifecycle with Autoregressive logic.

    Features:
    - Mixed Precision Training (AMP).
    - Early Stopping.
    - Scheduled Sampling: Gradually transitions from Teacher Forcing to Autoregressive inference.
      This helps the model learn to recover from its own past errors (mitigating Exposure Bias).

    Args:
        model (nn.Module): The neural network architecture.
        loss_fn (Callable): The function to compute the loss (e.g., Combined Loss).
        optimizer (optim.Optimizer): The optimization algorithm.
        device (torch.device): Compute device (CPU or CUDA).
        train_loader (DataLoader): Loader for training data.
        val_loader (DataLoader): Loader for validation data.
        test_loader (DataLoader): Loader for test data.
        n_future (int): Number of frames to predict into the future during training.
        early_stopping_patience (int): Epochs to wait without improvement.
    """
    def __init__(
        self, 
        model: nn.Module, 
        loss_fn: Callable, 
        optimizer: optim.Optimizer, 
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        n_future: int,
        early_stopping_patience: int = 10
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.n_future = n_future
        
        self.scaler = torch.amp.GradScaler('cuda')
        self.patience = early_stopping_patience
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=3, factor=0.5)
        
        # Scheduled Sampling Parameters
        self.teacher_forcing_ratio = 1.0  # Start fully supervised
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_wts = copy.deepcopy(self.model.state_dict())

    def _update_teacher_forcing_ratio(self, epoch: int, total_epochs: int):
        """
        Linearly decays the teacher forcing ratio from 1.0 to 0.0 over the course of training.
        
        Ratio = 1.0 -> Always use Ground Truth (Teacher Forcing).
        Ratio = 0.0 -> Always use Model Prediction (Autoregressive).
        """
        # Simple linear decay strategy
        self.teacher_forcing_ratio = max(0.0, 1.0 - (epoch / float(total_epochs)))

    def _train_one_epoch(self, epoch_index: int, total_epochs: int) -> float:
        self.model.train()
        total_loss = 0.0
        
        # Update sampling strategy for this epoch
        self._update_teacher_forcing_ratio(epoch_index, total_epochs)
        
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch_index}/{total_epochs} [TF: {self.teacher_forcing_ratio:.2f}]", leave=False)

        for X_batch, y_batch in loop:
            # X_batch: (B, T_past, 1, H, W) -> Initial Context
            # y_batch: (B, T_future, 1, H, W) -> Ground Truth Target Sequence
            
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # This tensor will hold the sequence of inputs for the model.
            # It starts as the initial past frames.
            current_input_seq = X_batch 
            
            # List to store the predictions for each future step
            predictions = []

            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                
                # Autoregressive Loop
                for t in range(self.n_future):
                    # Predict the next frame
                    # Model expects (B, T, C, H, W) and outputs (B, 1, 1, H, W)
                    y_pred_step = self.model(current_input_seq)
                    predictions.append(y_pred_step)
                    
                    # Scheduled Sampling Logic (The Decision)
                    # Should we feed the Truth or the Prediction to the next step?
                    
                    # Get the ground truth for this specific step 't'
                    # y_batch is (B, T_fut, 1, H, W) -> Slice to (B, 1, 1, H, W)
                    truth_step = y_batch[:, t:t+1, :, :, :]
                    
                    use_truth = random.random() < self.teacher_forcing_ratio
                    
                    next_input_frame = truth_step if use_truth else y_pred_step

                    # Update the sliding window
                    # Remove the oldest frame and append the new one
                    current_input_seq = torch.cat([current_input_seq[:, 1:, :, :, :], next_input_frame], dim=1)

                # Loss Calculation
                # Stack predictions along the time dimension: List -> (B, T_fut, 1, H, W)
                y_pred_full_seq = torch.stack([p.squeeze(1) for p in predictions], dim=1)
                
                # Calculate loss against the full sequence
                loss = self.loss_fn(y_pred_full_seq, y_batch)

            # Backward & Optimize
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(self.train_loader)

    def _validate_one_epoch(self) -> float:
        """
        Validation is ALWAYS fully autoregressive (Teacher Forcing = 0).
        We want to know how the model performs in the real world.
        """
        self.model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for X_val, y_val in self.val_loader:
                X_val = X_val.to(self.device)
                y_val = y_val.to(self.device)
                
                # Validation Logic: Predict n_future steps purely autoregressively
                current_input_seq = X_val
                predictions = []
                
                with torch.amp.autocast('cuda'):
                    for _ in range(self.n_future):
                        y_pred_step = self.model(current_input_seq)
                        predictions.append(y_pred_step)
                        
                        # Always use own prediction for next step
                        current_input_seq = torch.cat([current_input_seq[:, 1:], y_pred_step], dim=1)
                    
                    y_pred_full_seq = torch.stack([p.squeeze(1) for p in predictions], dim=1)
                    val_loss = self.loss_fn(y_pred_full_seq, y_val)
                
                running_val_loss += val_loss.item()
                
        return running_val_loss / len(self.val_loader)

    def train(self, num_epochs: int):
        print(f"Starting training on {self.device} for {num_epochs} epochs.")
        print(f"Future prediction horizon: {self.n_future} frames.")
        
        start_time = time.time()
        epochs_no_improve = 0

        for epoch in range(1, num_epochs + 1):
            train_loss = self._train_one_epoch(epoch, num_epochs)
            val_loss = self._validate_one_epoch()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
                save_msg = "(*)"
            else:
                epochs_no_improve += 1
                save_msg = ""

            print(f"Epoch [{epoch}/{num_epochs}] {save_msg} "
                  f"| Train: {train_loss:.5f} "
                  f"| Val: {val_loss:.5f} "
                  f"| TF-Ratio: {self.teacher_forcing_ratio:.2f}")

            if epochs_no_improve >= self.patience:
                print(f"\nEarly stopping triggered.")
                break

        self.model.load_state_dict(self.best_model_wts)
        print("Training complete. Best model loaded.")

    def test(self) -> float:
        # Re-uses the validation logic (Fully Autoregressive)
        self.model.eval()
        test_loss = 0.0
        print("\nStarting evaluation on Test Set (Autoregressive)...")
        
        with torch.no_grad():
            for X_test, y_test in self.test_loader:
                X_test = X_test.to(self.device)
                y_test = y_test.to(self.device)
                
                current_input_seq = X_test
                predictions = []
                
                with torch.amp.autocast('cuda'):
                    for _ in range(self.n_future):
                        y_pred_step = self.model(current_input_seq)
                        predictions.append(y_pred_step)
                        current_input_seq = torch.cat([current_input_seq[:, 1:], y_pred_step], dim=1)
                    
                    y_pred_full_seq = torch.stack([p.squeeze(1) for p in predictions], dim=1)
                    loss = self.loss_fn(y_pred_full_seq, y_test)
                
                test_loss += loss.item()
                
        avg_loss = test_loss / len(self.test_loader)
        print(f"Test Set Average Loss: {avg_loss:.6f}")
        return avg_loss
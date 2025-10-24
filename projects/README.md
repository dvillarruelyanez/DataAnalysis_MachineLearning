# 🧠 Research Projects – Data Science & Machine Learning (Master’s in Physics)

This folder contains six collaborative research projects forming the basis of two joint scientific papers developed during the course *PHYMSCFUN02 – Data Science and Machine Learning*.

Each project folder (`project01/` to `project06/`) includes:
- `data/`: datasets or preprocessing scripts
- `src/`: source code for models, loss functions, utilities
- `notebooks/`: exploratory analysis, experiments, and final results

---

## 📄 Joint Paper 1  
**Title:**  
**Decoding Collective Dynamics: Machine Learning Insights into Active Matter Simulations**

### 📦 Dataset
**The Well Dataset** – Simulations of active rod-like particles in a fluid medium:
- High-resolution spatial-temporal data: 81 time steps, 256×256 grid
- Multiple fields: scalar, vector, tensor
- Systematic variation of simulation parameters (alignment, dipole strength)
- Unified processing: normalization, masking, alignment

---

### 🔬 Project 01 – Predicting Emergent Dynamics (Team 1)

**Goal:**  
Understand how collective behaviors in active matter evolve over time and assess whether short-term dynamics can be forecasted directly from past observations.

**Method:**  
Apply deep sequence modeling approaches such as recurrent neural networks (RNNs) and attention-based models to global observables extracted from the simulation data (e.g., vorticity, scalar order parameters, energy distribution).

**Analysis:**  
Evaluate the accuracy of multi-step predictions across different dynamical regimes. Investigate error accumulation, model stability, and the limits of predictability depending on system parameters.

**Output:**  
Forecasted trajectories of global observables with quantified error trends across parameter regimes, highlighting regions of predictability and chaos.

---

### 🔬 Project 02 – Revealing Hidden Order (Team 2)

**Goal:**  
Uncover low-dimensional latent structures that capture the essential features and transitions in the spatiotemporal dynamics of active matter systems.

**Method:**  
Use autoencoders and variational autoencoders (VAEs) to learn compact latent representations from individual simulation frames or trajectories, trained across multiple regimes.

**Analysis:**  
Examine whether the latent spaces reflect known physical transitions (e.g., isotropic to nematic), group different regimes meaningfully, and organize the dynamics along interpretable axes (e.g., alignment, dipole strength).

**Output:**  
Latent space visualizations showing clustering by physical phase or parameter regime, and smooth trajectories over time reflecting system evolution.

---

### 🔬 Project 03 – Learning Physics-Aware Surrogates (Team 3)

**Goal:**  
Build generalizable and interpretable models that combine machine learning with known physical constraints to simulate active matter dynamics across parameter spaces.

**Method:**  
Implement hybrid models that integrate data-driven learning with known symmetries, conservation laws, or coarse-grained physics (e.g., PINNs or rule-based neural surrogates). Incorporate structures such as divergence-free fields, symplectic updates, or PDE residuals.

**Analysis:**  
Assess the generalization of models to unseen parameter regimes, evaluate robustness to noise, and compare against pure ML baselines. Examine whether physical constraints improve extrapolation and sample efficiency.

**Output:**  
Physics-informed surrogate models that demonstrate better transferability and alignment with known physical laws, supported by performance plots and qualitative trajectory comparisons.

---

## 📄 Joint Paper 2  
**Title:**  
**From Images to Equations: Machine Learning Models of Zebrafish Morphogenesis from Brightfield Time-Lapse Data**

### 📦 Dataset
- Brightfield time-lapse videos from 150 zebrafish embryos (normal, Nodal mutant, BMP mutant)
- Time range: 2–16 hpf, one frame every 5 minutes
- Pre-segmented masks of embryo regions available
- Preprocessing: resizing, normalization, alignment
- Extracted descriptors: projected area, major/minor axis lengths, optical flow fields
- Optional: Drosophila and Xenopus time-lapse datasets

---

### 🔬 Project 04 – Latent Mapping of Developmental Trajectories (Team 4)

**Goal:**  
Extract low-dimensional representations that organize morphological changes in time and across genotypes.

**Method:**  
- Train autoencoders or VAEs on raw or masked images  
- Track latent trajectories over time  
- Optionally reduce further via t-SNE or PCA  

**Analysis:**  
- Do latent spaces reflect biological time or known developmental transitions (blastula → gastrula)?  
- Can different genotypes be separated in latent space?  
- Are developmental stages aligned across embryos?  

**Output:**  
Latent trajectory plots, cluster separability, and genotype comparisons.  
**Key Insight:** Latent representations reveal structure in morphogenesis—smooth developmental progressions and potential markers of mutation effects.

---

### 🔬 Project 05 – Forecasting Morphogenesis in Latent and Image Space (Team 5)

**Goal:**  
Predict embryo morphology at future time points.

**Method:**  
- Encode images into latent space (from Project 4)  
- Use temporal models (ConvLSTM, temporal Transformer) to forecast future latent states  
- Decode back to image domain using trained decoder  

**Analysis:**  
- Evaluate structural similarity (SSIM, PSNR) with real images  
- Measure biological metrics such as projected area, aspect ratio, and centroid motion  

**Output:**  
Forecasted image sequences, error curves, and genotype comparisons.  
**Key Insight:** ML models can anticipate early morphogenetic events—potentially useful for detecting abnormal development.

---

### 🔬 Project 06 – Physics-Informed Neural Surrogates for Morphogenesis (Team 6)

**Goal:**  
Bridge data-driven embeddings with mechanistic models via physics-informed neural networks (PINNs).

**Method:**  
- Use latent coordinates and derived quantities (area, curvature, optical flow) to define spatiotemporal fields  
- Fit PINNs enforcing PDE residuals corresponding to plausible physical models (e.g., curvature-driven flow, viscoelastic deformation)  
- Total loss = data mismatch + PDE residuals  

**Analysis:**  
- Does the PINN reproduce known dynamics in control embryos?  
- Are inferred coefficients biologically meaningful (e.g., growth rate, tension)?  
- Can the model generalize across genotypes?  

**Output:**  
PDE fits, residual maps, physical parameter estimates.  
**Key Insight:** ML and physics can be fused to create interpretable, generative models of embryonic shape change.

---

## 📁 Folder Structure per Project

Each `projectXX/` folder follows this structure:
```
projectXX/
├── data/                # raw/processed data (or symlinks)
├── notebooks/           # exploratory & analysis notebooks
├── src/                 # reusable code (dataloaders, models, utils)
├── models/              # saved checkpoints
├── results/             # figures, tables, metrics
├── reports/             # week summaries, project plans
├── requirements.txt     # pinned deps (or environment.yml)
└── README.md
```

---

## 📚 Collaboration & Publishing

- Final manuscripts prepared collaboratively in Overleaf  
- Code and figures must be fully reproducible from this repository  
- Intended for arXiv submission and journal review  
- Supplementary material will be uploaded to Zenodo

---

**Instructor:** Dr. Hernán Andrés Morales-Navarrete  
**Course:** Master's in Physics – Data Science and Machine Learning (PHYMSCFUN02)  
**Semester:** Second Semester 2025  


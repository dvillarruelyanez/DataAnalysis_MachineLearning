# Joint Paper 1  
**Title:** *Decoding Collective Dynamics: Machine Learning Insights into Active Matter Simulations*  
**Dataset:** **The Well Dataset** — simulations of active rod-like particles (81 time steps, 256×256 grid; scalar/vector/tensor fields; parameter sweeps for alignment & dipole strength; unified processing).

## Project 03

**Goal:** Build interpretable, generalizable surrogates that integrate **physical constraints** (e.g., divergence-free fields, conservation) with ML to simulate active matter across parameter regimes.

### Activities 
- **Physics characterization**
  - Identify key constraints/symmetries (e.g., incompressibility, energy budget).  
  - Express them mathematically; map each to a viable **loss term**, architectural constraint, or training prior.  
  - List candidate PDEs/residuals relevant to the dataset and feasible to include.
- **Baseline emulation**  
  - Train a simple next-step predictor (e.g., shallow CNN or UNet-lite) from past frames/fields.  
  - Record accuracy and **failure modes** (e.g., drift, non-physical artifacts).
- **Hybrid model plan**  
  - Survey PINNs / physics-guided networks for fluid/active matter.  
  - Draft a **hybrid approach**: physics loss (PDE residuals), divergence-free projection layers, or symplectic/volume-preserving updates.  
  - Define evaluation metrics: physical residuals, divergence norm, generalization to unseen parameters, robustness to noise.

### Deliverables 
- Notebook: preprocessing pipeline & PCA/t-SNE  
- Draft architecture & evaluation plan

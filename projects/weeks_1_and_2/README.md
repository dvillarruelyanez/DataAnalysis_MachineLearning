# Course Research Projects — Week 1 Guide

This README provides **Week 2 plans**  for two joint research papers. Each team has **2 students** working **~20 hours/week**. Weeks 1 and 2 focuses on **foundations, dataset understanding, EDA, baselines, and a concrete model plan**.

---

## Repository Structure (suggested)

```
.
├── data/                # raw/processed data (or symlinks)
├── notebooks/           # exploratory & analysis notebooks
├── src/                 # reusable code (dataloaders, models, utils)
├── models/              # saved checkpoints
├── results/             # figures, tables, metrics
├── reports/             # week summaries, project plans
├── requirements.txt     # pinned deps (or environment.yml)
└── README.md
```
## Week 1-2 Objective
Establish scientific framing, ensure a reproducible setup, and perform initial EDA to de-risk modeling in Weeks 2–6.

### For All Teams (Joint Prep)
- [ ] **Scientific context & scope**  
  - Read/align on paper outline, target questions, expected contributions.  
  - Draft **1-page project proposal**: problem, approach, hypotheses, integration into joint manuscript.
- [ ] **Environment & data readiness **  
  - Create GitHub repo & structure (`data/`, `notebooks/`, `src/`, `reports/`).  
  - Reproducible env (Conda + `requirements.txt` or `environment.yml`).  
  - Load and inspect **The Well Dataset** (shapes, fields, params).  
  - Write a **dataset summary report** (variables, preprocessing, gaps).
---

# Joint Paper 1  
**Title:** *Decoding Collective Dynamics: Machine Learning Insights into Active Matter Simulations*  
**Dataset:** **The Well Dataset** — simulations of active rod-like particles (81 time steps, 256×256 grid; scalar/vector/tensor fields; parameter sweeps for alignment & dipole strength; unified processing).

**References:**
- The Well: a Large-Scale Collection of Diverse Physics Simulations for Machine Learning: https://arxiv.org/abs/2412.00568
- Learning fast, accurate, and stable closures of a kinetic theory of an active fluid: https://www.sciencedirect.com/science/article/pii/S0021999124001189 
- The Well GitHub repository: https://github.com/PolymathicAI/the_well/blob/master/docs/tutorials/dataset.ipynb
---

## Project 01 — Predicting Emergent Dynamics (Team 1: Jorge and Jhon)
**Goal:** Forecast short-term evolution of **global observables** (e.g., vorticity, order parameters, energy).

### Activities 
- [ ] **Feature extraction & baseline EDA **  
  - Compute time-series of targeted observables; visualize trends & autocorrelation.
- [ ] **Problem framing & baseline model **  
  - Fix input window & forecast horizon (e.g., 5→5 steps).  
  - Baselines: persistence, linear regression; evaluate MSE/MAE.
- [ ] **Advanced model plan **  
  - Literature skim on RNN/LSTM/attention for physics time series.  
  - Draft **model spec** (I/O shapes, loss, training protocol, validation split).

### Deliverables 
- [ ] Notebook: data exploration & observable extraction  
- [ ] Baseline prediction results + plots  
- [ ] **2–3 page model plan** (architecture, metrics, schedule)

---

## Project 02 — Revealing Hidden Order (Team 2: Leonel and Isaac)
**Goal:** Learn **low-dimensional latent spaces** that capture phases/transitions and organize dynamics across regimes.

### Activities 
- [ ] **Preprocessing & sampling **  
  - Decide representation (frames, sequences, patches).  
  - Normalize/standardize; create a prototype subset (e.g., 500–1000 frames).
- [ ] **Exploratory visualization **  
  - PCA / t-SNE on flattened frames or descriptors; inspect phase/regime separation.
- [ ] **Latent model design **  
  - AE/VAE architecture sketch (encoder/decoder, latent dims, losses).  
  - Define metrics: clustering separation, reconstruction error, trajectory smoothness.


## Project 03 — Learning Physics-Aware Surrogates (Team 3: Franklin and Daniel)
**Goal:** Build interpretable, generalizable surrogates that integrate **physical constraints** (e.g., divergence-free fields, conservation) with ML to simulate active matter across parameter regimes.

### Activities 
- [ ] **Physics characterization **  
  - Identify key constraints/symmetries (e.g., incompressibility, energy budget).  
  - Express them mathematically; map each to a viable **loss term**, architectural constraint, or training prior.  
  - List candidate PDEs/residuals relevant to the dataset and feasible to include.
- [ ] **Baseline emulation **  
  - Train a simple next-step predictor (e.g., shallow CNN or UNet-lite) from past frames/fields.  
  - Record accuracy and **failure modes** (e.g., drift, non-physical artifacts).
- [ ] **Hybrid model plan **  
  - Survey PINNs / physics-guided networks for fluid/active matter.  
  - Draft a **hybrid approach**: physics loss (PDE residuals), divergence-free projection layers, or symplectic/volume-preserving updates.  
  - Define evaluation metrics: physical residuals, divergence norm, generalization to unseen parameters, robustness to noise.



### Deliverables 
- [ ] Notebook: preprocessing pipeline & PCA/t-SNE  
- [ ] Draft architecture & evaluation plan

---

## Paper 1 — End-of-Week Milestone (All Teams)
- [ ] GitHub repo initialized with working data loaders & EDA notebooks  
- [ ] **Short project plan** (2–3 pages) with goals, hypotheses, approach  
- [ ] **Timeline proposal** for Weeks 2–6 (key experiments & checkpoints)

---

# Joint Paper 2  
**Title:** *From Images to Equations: Machine Learning Models of Zebrafish Morphogenesis from Brightfield Time-Lapse Data*  
**Dataset:** Brightfield time-lapse videos of **150 zebrafish embryos** (normal, Nodal mutant, BMP mutant), **2–16 hpf**, 1 frame/5 min. Pre-segmented masks; preprocessing (resizing, normalization, alignment). Extracted descriptors (area, axes lengths, optical flow). *(Optional: Drosophila/Xenopus)*

**References:**
- Review developmental stages (blastula → gastrula), expected morphology, mutant phenotypes:
- Stages of embryonic development of the zebrafish: https://anatomypubs.onlinelibrary.wiley.com/doi/10.1002/aja.1002030302
- EmbryoNet: using deep learning to link embryonic phenotypes to signaling pathways: https://www.nature.com/articles/s41592-023-01873-4


---

## Project 04 — Latent Mapping of Developmental Trajectories (Team 4: Pablo)
**Goal:** Learn latent spaces that organize **time** and **genotype** and align developmental stages across embryos.

### Activities 
- [ ] **QC & descriptors **  
  - Load sequences for all genotypes; check alignment/masks.  
  - Plot descriptors over time (area, major/minor axis, aspect ratio).
- [ ] **Baseline DR **  
  - PCA / t-SNE on images or descriptors; test grouping by stage/genotype.
- [ ] **Latent model plan **  
  - AE/VAE prototype (input shape, latent size, losses).  
  - Plan trajectory visualization (per-embryo paths through latent space).  
  - Metrics: stage separability, genotype clustering, reconstruction error.

### Deliverables 
- [ ] Notebook: QC + descriptor trends + PCA/t-SNE  
- [ ] Draft AE/VAE architecture & evaluation criteria

---

## Project 05 — Forecasting Morphogenesis (Team 5: Alejandro and Angel)
**Goal:** Forecast future embryo morphology in **latent** and **image space**.

### Activities 
- [ ] **Data pipeline **  
  - Dataloaders for sequences + masks; embryo-wise or genotype-wise splits.  
  - Placeholder encoder/decoder (or use descriptors initially).
- [ ] **Temporal EDA **  
  - Time evolution of area, centroid motion, axis lengths; auto/cross-correlations.
- [ ] **Forecasting design **  
  - Choose input window & horizon (e.g., 6 frames ≈ 30 min ahead).  
  - Draft ConvLSTM / temporal Transformer plan; losses & evaluation  
    (SSIM, PSNR, RMSE on biological features; genotype-stratified metrics).

### Deliverables 
- [ ] Notebook: temporal descriptor analysis  
- [ ] Baseline: **persistence** forecaster & error curves  
- [ ] Draft architecture & training plan

---

## Paper 2 — End-of-Week Milestone (All Teams)
- [ ] Working EDA notebooks (loading, QC, plots)  
- [ ] **Modeling plan** (architecture, metrics, training schedule)  


---

## Reproducibility & Collaboration Checklist
- [ ] Pin dependencies (`requirements.txt` / `environment.yml`).  
- [ ] Use **.env** / config files for paths; avoid hard-coding.  
- [ ] Commit notebooks with **outputs cleared**; save plots to `results/`.  
- [ ] Open an **Issues** thread per team with Week 1 deliverables & blockers.  
- [ ] Use **Pull Requests** for code review between teammates.

---

## Submission (End of Week 1)
Upload to repo:
- `reports/ProjectPlan_TeamX.pdf` (2–3 pages)  
- `notebooks/` (EDA + baselines)  
- `results/figures/` (key plots)  
- `README.md` (this file)

> **Tip:** Keep figures informative and minimal: axis labels, units, genotype/phase legends, and clear captions.

# Course Research Projects — Week 1 Guide

This README provides **Week 1 plans** (of 6) for two joint research papers. Each team has **2 students** working **~20 hours/week**. Week 1 focuses on **foundations, dataset understanding, EDA, baselines, and a concrete model plan**.

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

---

# Joint Paper 1  
**Title:** *Decoding Collective Dynamics: Machine Learning Insights into Active Matter Simulations*  
**Dataset:** **The Well Dataset** — simulations of active rod-like particles (81 time steps, 256×256 grid; scalar/vector/tensor fields; parameter sweeps for alignment & dipole strength; unified processing).

## Week 1 Objective
Establish scientific framing, ensure a reproducible setup, and perform initial EDA to de-risk modeling in Weeks 2–6.

### For All Teams (Joint Prep, ~6h)
- [ ] **Scientific context & scope (2h)**  
  - Read/align on paper outline, target questions, expected contributions.  
  - Draft **1-page project proposal**: problem, approach, hypotheses, integration into joint manuscript.
- [ ] **Environment & data readiness (4h)**  
  - Create GitHub repo & structure (`data/`, `notebooks/`, `src/`, `reports/`).  
  - Reproducible env (Conda + `requirements.txt` or `environment.yml`).  
  - Load and inspect **The Well Dataset** (shapes, fields, params).  
  - Write a **dataset summary report** (variables, preprocessing, gaps).

---

## Project 01 — Predicting Emergent Dynamics (Team 1)
**Goal:** Forecast short-term evolution of **global observables** (e.g., vorticity, order parameters, energy).

### Activities (~14h)
- [ ] **Feature extraction & baseline EDA (6h)**  
  - Compute time-series of targeted observables; visualize trends & autocorrelation.
- [ ] **Problem framing & baseline model (4h)**  
  - Fix input window & forecast horizon (e.g., 5→5 steps).  
  - Baselines: persistence, linear regression; evaluate MSE/MAE.
- [ ] **Advanced model plan (4h)**  
  - Literature skim on RNN/LSTM/attention for physics time series.  
  - Draft **model spec** (I/O shapes, loss, training protocol, validation split).

### Deliverables (end of Week 1)
- [ ] Notebook: data exploration & observable extraction  
- [ ] Baseline prediction results + plots  
- [ ] **2–3 page model plan** (architecture, metrics, schedule)

---

## Project 02 — Revealing Hidden Order (Team 2)
**Goal:** Learn **low-dimensional latent spaces** that capture phases/transitions and organize dynamics across regimes.

### Activities (~14h)
- [ ] **Preprocessing & sampling (6h)**  
  - Decide representation (frames, sequences, patches).  
  - Normalize/standardize; create a prototype subset (e.g., 500–1000 frames).
- [ ] **Exploratory visualization (4h)**  
  - PCA / t-SNE on flattened frames or descriptors; inspect phase/regime separation.
- [ ] **Latent model design (4h)**  
  - AE/VAE architecture sketch (encoder/decoder, latent dims, losses).  
  - Define metrics: clustering separation, reconstruction error, trajectory smoothness.

### Deliverables (end of Week 1)
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

## Week 1 Objective
Ground biological context, validate data quality, produce descriptor/latent EDA, and fix an initial modeling plan.

### For All Teams (Joint Prep, ~6h)
- [ ] **Biology & manuscript framing (2h)**  
  - Review developmental stages (blastula → gastrula), expected morphology, mutant phenotypes.  
  - Draft **1-page research summary** (goals, hypotheses, contribution).
- [ ] **Tech setup & dataset scan (4h)**  
  - Repo structure (`data/`, `notebooks/`, `models/`, `results/`).  
  - Verify libraries (PyTorch/TensorFlow, OpenCV, scikit-image/learn, matplotlib).  
  - **Dataset report**: counts, resolutions, masks, descriptors, splits.

---

## Project 04 — Latent Mapping of Developmental Trajectories (Team 4)
**Goal:** Learn latent spaces that organize **time** and **genotype** and align developmental stages across embryos.

### Activities (~14h)
- [ ] **QC & descriptors (4h)**  
  - Load sequences for all genotypes; check alignment/masks.  
  - Plot descriptors over time (area, major/minor axis, aspect ratio).
- [ ] **Baseline DR (4h)**  
  - PCA / t-SNE on images or descriptors; test grouping by stage/genotype.
- [ ] **Latent model plan (6h)**  
  - AE/VAE prototype (input shape, latent size, losses).  
  - Plan trajectory visualization (per-embryo paths through latent space).  
  - Metrics: stage separability, genotype clustering, reconstruction error.

### Deliverables (end of Week 1)
- [ ] Notebook: QC + descriptor trends + PCA/t-SNE  
- [ ] Draft AE/VAE architecture & evaluation criteria

---

## Project 05 — Forecasting Morphogenesis (Team 5)
**Goal:** Forecast future embryo morphology in **latent** and **image space**.

### Activities (~14h)
- [ ] **Data pipeline (4h)**  
  - Dataloaders for sequences + masks; embryo-wise or genotype-wise splits.  
  - Placeholder encoder/decoder (or use descriptors initially).
- [ ] **Temporal EDA (4h)**  
  - Time evolution of area, centroid motion, axis lengths; auto/cross-correlations.
- [ ] **Forecasting design (6h)**  
  - Choose input window & horizon (e.g., 6 frames ≈ 30 min ahead).  
  - Draft ConvLSTM / temporal Transformer plan; losses & evaluation  
    (SSIM, PSNR, RMSE on biological features; genotype-stratified metrics).

### Deliverables (end of Week 1)
- [ ] Notebook: temporal descriptor analysis  
- [ ] Baseline: **persistence** forecaster & error curves  
- [ ] Draft architecture & training plan

---

## Paper 2 — End-of-Week Milestone (All Teams)
- [ ] Working EDA notebooks (loading, QC, plots)  
- [ ] **Modeling plan** (architecture, metrics, training schedule)  
- [ ] **Weeks 2–6 timeline** per team

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

# Reconstruction-attack pipeline — investigation findings

Record of what was tried to raise the end-to-end leak quality (fraction of cid-0
images reconstructed at SSIM > 0.7), what worked, what didn't, and why. Goal: don't
re-walk the dead ends.

Baseline at the start of the investigation: **21.4%** SSIM > 0.7.

## TL;DR

- The bottleneck is the **reconstruction model's train→real domain gap**, NOT the
  clustering. Perfect (oracle) clustering lifts SSIM>0.7 only ~24→28%.
- **Worked:** randomizing the per-step view count in training (model was OOD on the
  small real clusters). 21.4% → ~24%.
- **Didn't work (reverted):** post-hoc cluster *merge* in `regroup_fragments.py`;
  *gaussian* corruption domain-randomization in `reconstruction_test.py`.
- **Next:** faithful corruption simulation (run the real LOKI leak on surrogate data)
  so training fragments carry the real degradation type, not gaussian noise.

## Diagnostic tooling built (kept)

- `cluster_purity.py` — EVALUATOR-only oracle (uses cid-0 ground truth like
  `compare_cid0.py`, never feeds the pipeline). Labels each clustered fragment by
  matching it to the known cid-0 images with two matchers: NCC (clustering-
  independent, ~15% view→source, noisy) and the BYOL encoder (~90%, trustworthy).
  Reports per-cluster instance/class purity, blend rate, over-split, coverage.

## Key findings

1. **Clustering is messy but not the bottleneck.** Direct purity (BYOL): ~73%
   size-weighted purity, **43.5% over-split** (mean 2.05 clusters/original), **~16%
   blends** (concentrated in big ≥60-view clusters), median **15 views/cluster** vs
   the ~54 perfect grouping would give.
2. **Oracle-clustering ceiling ≈ 28%.** Grouping every usable fragment (incl. the
   ~515k DBSCAN discarded as noise) by true BYOL-matched source, cap 99 views, only
   reaches 27.8% >0.7 (vs 24% real). It *does* help looser bars a lot (>0.5: 50→65%,
   recognizable: 76→88%). So clustering work is low-leverage for the >0.7 metric.
3. **The model is the wall.** On *clean* held-out views the model does **78% >0.7 at
   54 views** (84% at 99). On *real* fragments at the same 54 views: **28%**. Same
   model, same view count — the gap is real-fragment corruption the training never saw.
4. **Model ↔ clustering are coupled.** A stronger model only pays off on good
   clusters; the over-split/blends are masked today because the model caps everything.

## Attempts

### ✅ Per-step view-count randomization (kept)
`reconstruction_test.py`: train on a log-uniform view count in [VIEWS_MIN, VIEWS_MAX]
each step instead of a fixed 64. Real clusters are median ~15 views; the fixed-64
model was OOD on them. **21.4% → ~24%.**

### ❌ Cluster merge (removed from `regroup_fragments.py`)
Post-`refine` agglomerative merge of over-split clusters via mutual-NN of BYOL
centroids (size-capped). **Removed — it cannot work.** Post-mortem: same-original
sibling clusters have median centroid cosine only **0.745** (0.7% exceed 0.90), and a
cluster's nearest neighbour is its true sibling just **16.4%** of the time. Same-class
*different* originals sit closer than same-original halves, so any blind centroid
merge is ~84% wrong. (`EPS_SIM` tuning is also a dead end — it's pinned at the giant-
component cliff ~0.89–0.90; 0.89 vs 0.90 is flat on every metric.)

### ❌ Gaussian corruption domain-randomization (removed from `reconstruction_test.py`)
Calibrated real corruption to ≈ σ0.17 additive (the level that drops the clean model
78%→28% @54 views), then retrained with per-step `LEAK_NOISE ~ U[0,0.2]`,
`LEAK_CONTAM_P ~ U[0,0.25]`. **Reverted — net regression on the real pipeline:**

| clusters | clean model | corruption model | Δ |
|---|---|---|---|
| real (median 15 views) | **24.1%** | 21.1% | −3.0 |
| oracle (~54 views) | 27.8% | **32.5%** | +4.7 |

It made a better *many-view* reconstructor (oracle gain confirms the domain gap is
real and movable) but a worse *few-view* one — it over-smooths starved real clusters.
Two causes: (1) gaussian noise ≠ real degradation (real = binning **blends** +
few-view + structured multi-epoch leak noise); (2) `evaluate()` selects the checkpoint
on a full-~100-view eval, biasing to the many-view regime. The corruption-trained
checkpoint is preserved as `reconstruction_out/best_leak_corruptrand.pt`; the active
`best_leak.pt` is the clean 24% model (`best_leak_viewsonly_24pct.pt`).

## Recommended next step

**Route 2 — faithful corruption simulation.** The corruption is a property of the
attack *mechanism* (server-controlled), not the client data, so generate training
pairs by running the real LOKI leak on *surrogate* images: push them through the trap
+ the configured local-epoch BYOL training (use the real round-R global encoder as the
downstream loss), reconstruct fragments via Eq. 9, pair with the clean originals.
Those fragments carry the real degradation type (blends/few-view/structured), unlike
gaussian noise. `attacks.py` already has the pieces (`Loki.simulate_client_update`,
`reconstruct`). Validate that simulated-fragment statistics match the real fragments
(we hold the real fragments, just no labels) *before* committing to a retrain.

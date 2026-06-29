"""
cluster_purity.py -- EVALUATOR-ONLY diagnostic: how clean is regroup_fragments'
*blind* clustering, scored against cid-0 ground truth.

THREAT-MODEL NOTE (read this before changing anything):
  This is an oracle the *researcher* runs to grade the attack, exactly like
  compare_cid0.py. It uses client 0's real images -- which the attack pipeline
  never sees -- only to LABEL each already-clustered fragment with its true source
  original, then measures per-cluster purity. NOTHING here feeds back into
  regroup_fragments / infer_fragments; those stay blind (unknown class count, no
  ground truth). Do not import anything from this file into the pipeline.

What it answers (the regroup failure mode is one of two, with opposite fixes):
  * BLEND   -- multiple originals land in one cluster, so the reconstructor is fed
               a superposition. Symptom: low instance purity, esp. in big clusters.
               Fix: tighten clustering (REFINE_CAP -> ~99, smaller EPS_SIM).
  * OVERSPLIT- one original is scattered across many clusters, so each cluster has
               too few views (the model trained on ~64-100). Symptom: high purity
               but high fragments-per-original and small clusters.
               Fix: looser clustering / train the model for small view sets.

Matching reliability: a clustered fragment is a heavy-crop/jitter VIEW, so matching
it to its 32x32 source by NCC is itself noisy. We therefore
  (a) CALIBRATE the matcher on known augmented views of cid-0 originals (top-1
      source-recovery accuracy) -- a PURE cluster can only score ~this purity, so
      every number below is read against that ceiling; and
  (b) lean on per-cluster vote STRUCTURE (one dominant peak = pure even if the
      noise floor scatters the rest; two peaks = a real blend), not raw purity.

Run after regroup_fragments.py has produced fragments_clustered/.
"""
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as Fn
import torchvision
from torchvision.utils import make_grid, save_image

import reconstruction_test as R
from architectures import ResNet18Projv3

# ---------------------------------------------------------------------------- #
# Config (mirrors compare_cid0.py so the ground truth is recovered identically)
# ---------------------------------------------------------------------------- #
CLUSTER_DIR = "fragments_clustered"
CKPT        = "eval_model.pth"   # BYOL encoder (same one regroup_fragments clusters with)
DATA_DIR    = R.DATA_DIR
NUM_CLIENTS = 5
CLS_PER     = 2
SEED        = 12345
TARGET_CID  = 0
QA_OUT      = "cluster_purity_out"

BLANK_STD   = 0.05      # raw-space per-fragment std below this == blank (matches infer)
MATCH_BLUR  = 2         # ncc light blur 32->16 (matches compare_cid0)
CHUNK       = 4096

# Matcher calibration: how well NCC recovers a view's source original.
CALIB_ORIG  = 400       # cid-0 originals sampled for the calibration
CALIB_VIEWS = 25        # augmented views generated per calibration original

# A cluster's #2 instance must hold at least this SHARE (and >=2 votes) of the
# cluster to be called a real blend (vs. matcher-noise scatter).
BLEND_SHARE = 0.25
PURE_SHARE  = 0.80      # top-1 share at/above this == "pure" (relative to ceiling)
MIN_CLUSTER = 2         # clusters smaller than this are reported separately


# ---------------------------------------------------------------------------- #
# Ground-truth recovery + matching (lifted from compare_cid0.py for identical semantics)
# ---------------------------------------------------------------------------- #
def client_indices(targets, cid):
    """Exact train indices held by `cid` (replicates the dataloader partitioner)."""
    labels = np.asarray(targets)
    num_classes = len(np.unique(labels))
    shards_per_class = int(np.ceil(NUM_CLIENTS * CLS_PER / num_classes))
    all_shards, rng = [], np.random.default_rng(SEED)
    for cls in range(num_classes):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        sz, rem, pos = len(cls_idx) // shards_per_class, len(cls_idx) % shards_per_class, 0
        for i in range(shards_per_class):
            end = pos + sz + (1 if i < rem else 0)
            all_shards.append(cls_idx[pos:end].tolist()); pos = end
    np.random.default_rng(SEED).shuffle(all_shards)
    shards = all_shards[cid * CLS_PER:(cid + 1) * CLS_PER]
    return np.array([i for s in shards for i in s])


def to_unit(x):
    """Per-image min-max to [0,1] -- the canonical fragment->image map (attacks.Loki)."""
    lo = x.amin(dim=(-3, -2, -1), keepdim=True)
    hi = x.amax(dim=(-3, -2, -1), keepdim=True)
    return (x - lo) / (hi - lo + 1e-8)


def ncc_feats(imgs01, device, blur=MATCH_BLUR):
    """imgs [N,3,32,32] in [0,1] -> blurred, mean-subtracted, L2-normalized rows.
    Brightness/contrast/blur-robust cross-correlation features (compare_cid0).

    Clustering-INDEPENDENT oracle: honest about blends (won't hide same-metric
    merges) but noisy on heavy crops (~15% view->source top-1), so absolute
    instance purity is barely above its noise floor -- read it relatively."""
    x = Fn.avg_pool2d(imgs01.to(device), blur)      # 32 -> 16
    x = x.flatten(1)
    x = x - x.mean(1, keepdim=True)
    return Fn.normalize(x, dim=1)


_BYOL = {}
_MEAN = torch.tensor(R.CIFAR_MEAN).view(1, 3, 1, 1)
_STD  = torch.tensor(R.CIFAR_STD).view(1, 3, 1, 1)


def byol_feats(imgs01, device):
    """imgs [N,3,32,32] in [0,1] -> L2-normalized BYOL backbone embeddings.

    The augmentation-INVARIANT oracle: a view embeds near its clean source, so
    view->source top-1 is far higher than NCC -> trustworthy coverage/instance ID.
    CAVEAT: it is the SAME metric regroup clusters with, so its blind spot is
    regroup's blind spot -- it can UNDER-report same-class blends. Use it for
    coverage/fragmentation; cross-check blends against the independent NCC oracle."""
    if "net" not in _BYOL:
        sd = torch.load(CKPT, map_location="cpu")
        net = ResNet18Projv3()
        net.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("net.")})
        _BYOL["net"] = net.to(device).eval().requires_grad_(False)
        _BYOL["mean"], _BYOL["std"] = _MEAN.to(device), _STD.to(device)
    net = _BYOL["net"]
    x = (to_unit(imgs01.to(device)) - _BYOL["mean"]) / _BYOL["std"]
    with torch.autocast("cuda", enabled=device.type == "cuda"):
        return Fn.normalize(net.backbone(x), dim=1)


def best_match_cpu(feats_cpu, gallery_dev, device, chunk=CHUNK):
    """For each (CPU) feature row, best (score, gallery index) over the gallery."""
    n = feats_cpu.shape[0]
    sc = torch.empty(n)
    idx = torch.empty(n, dtype=torch.long)
    for s in range(0, n, chunk):
        m = feats_cpu[s:s + chunk].to(device).float() @ gallery_dev.float().t()
        mx = m.max(1)
        sc[s:s + chunk] = mx.values.cpu()
        idx[s:s + chunk] = mx.indices.cpu()
    return sc, idx


# ---------------------------------------------------------------------------- #
# Matcher calibration: view -> source recovery accuracy (the purity ceiling)
# ---------------------------------------------------------------------------- #
def calibrate_matcher(cid0_u8, gallery, feats_fn, device):
    """Generate augmented views of known cid-0 originals (the exact federated
    augmentation) and measure how often `feats_fn` matches a view back to its true
    source -- the ceiling a PURE cluster can reach. Returns top1 accuracy."""
    g = torch.Generator().manual_seed(SEED)
    samp = torch.randperm(cid0_u8.shape[0], generator=g)[:CALIB_ORIG].numpy()
    feats, src = [], []
    for gi in samp:
        views = R.pil_views(cid0_u8[gi], CALIB_VIEWS, seed=SEED * 131 + int(gi))  # [v,3,32,32] u8
        v01 = torch.from_numpy(views).float() / 255.0
        feats.append(feats_fn(v01, device).half().cpu())
        src.append(torch.full((CALIB_VIEWS,), int(gi), dtype=torch.long))
    feats = torch.cat(feats); src = torch.cat(src)
    _, idx = best_match_cpu(feats, gallery, device)
    return float((idx == src).float().mean())


# ---------------------------------------------------------------------------- #
# Load + match every clustered fragment, grouped by cluster
# ---------------------------------------------------------------------------- #
def load_and_embed(device):
    """One disk pass over every cluster -> both feature sets per fragment, plus the
    cluster id and size of each. Returns (ncc_feats, byol_feats, cl_of_frag, sizes)."""
    bins = sorted(glob.glob(os.path.join(CLUSTER_DIR, "bin_*")))
    f_ncc, f_byol, cl_of_frag, sizes = [], [], [], []
    print(f"embedding fragments in {len(bins)} clusters (NCC + BYOL)...")
    for bi, b in enumerate(bins):
        paths = sorted(glob.glob(os.path.join(b, "round_*.pt")))
        if not paths:
            sizes.append(0); continue
        frags = torch.stack([torch.load(p, map_location="cpu", weights_only=False) for p in paths])
        keep = frags.flatten(1).std(dim=1) >= BLANK_STD          # drop any blanks (redundant w/ regroup)
        if bool(keep.any()):
            frags = frags[keep]
        f_ncc.append(ncc_feats(to_unit(frags), device).half().cpu())
        f_byol.append(byol_feats(frags, device).half().cpu())
        cl_of_frag.append(torch.full((frags.shape[0],), bi, dtype=torch.long))
        sizes.append(frags.shape[0])
        if (bi + 1) % 2000 == 0:
            print(f"  {bi+1}/{len(bins)}", flush=True)
    return (torch.cat(f_ncc), torch.cat(f_byol),
            torch.cat(cl_of_frag), torch.tensor(sizes))


def per_cluster_stats(inst, cls, cl_of_frag, n_clusters):
    """Group fragment votes by cluster -> per-cluster vote structure."""
    order = torch.argsort(cl_of_frag)
    inst_s, cls_s, cl_s = inst[order], cls[order], cl_of_frag[order]
    # boundaries between clusters in the sorted array
    bounds = torch.searchsorted(cl_s, torch.arange(n_clusters + 1))
    out = []
    for bi in range(n_clusters):
        lo, hi = int(bounds[bi]), int(bounds[bi + 1])
        n = hi - lo
        if n == 0:
            out.append((0, 0.0, 0.0, -1, 0.0)); continue
        vi = inst_s[lo:hi]
        vc = cls_s[lo:hi]
        # instance vote structure
        u, c = torch.unique(vi, return_counts=True)
        topc, topi = c.max(0)
        top1 = int(topc); maj_inst = int(u[topi])
        top2 = int(c.sort(descending=True).values[1]) if c.numel() > 1 else 0
        # class purity
        _, cc = torch.unique(vc, return_counts=True)
        cls_pur = float(cc.max()) / n
        out.append((n, top1 / n, top2 / n, maj_inst, cls_pur))
    return out  # list of (size, top1_share, top2_share, maj_inst, class_purity)


def analyze(name, feats, gallery, top1, labels0, cl_of_frag, sizes, n_orig):
    """Match every fragment with one oracle, then report purity / blend / coverage."""
    n_clusters = len(sizes)
    _, inst = best_match_cpu(feats, gallery, R.get_device())
    cls = labels0[inst]
    stats = per_cluster_stats(inst, cls, cl_of_frag, n_clusters)
    S  = torch.tensor([s[0] for s in stats])
    P1 = torch.tensor([s[1] for s in stats])
    P2 = torch.tensor([s[2] for s in stats])
    MAJ = torch.tensor([s[3] for s in stats])
    CP = torch.tensor([s[4] for s in stats])
    big = S >= MIN_CLUSTER

    frag_in_maj = float((P1[big] * S[big]).sum() / S[big].sum())
    blend = big & (P2 >= BLEND_SHARE) & (P2 * S >= 2)
    pure  = big & (P1 >= PURE_SHARE * top1)

    print(f"\n############## ORACLE: {name}  (view->source ceiling {top1*100:.1f}%) ##############")
    print("---- INSTANCE PURITY ----")
    print(f"size-weighted purity (frags in cluster's top original): {frag_in_maj*100:5.1f}%  "
          f"-> {frag_in_maj/max(top1,1e-9)*100:.0f}% of ceiling")
    print(f"median top-1 share {float(P1[big].median())*100:.1f}%   "
          f"median class purity {float(CP[big].median())*100:.1f}%")
    print(f"clear BLEND (2nd original >= {int(BLEND_SHARE*100)}% & >=2 votes): "
          f"{int(blend.sum())}/{int(big.sum())} ({float(blend.float().mean())*100:.1f}%)   "
          f"PURE: {float(pure.float().mean())*100:.1f}%")

    print("---- size vs purity (blend should NOT rise with size if clustering is clean) ----")
    print("  size bucket     n        purity   blend%")
    edges = [2, 4, 6, 8, 12, 20, 35, 60, 100, 199]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = big & (S >= lo) & (S < hi)
        if bool(m.any()):
            pur = float((P1[m] * S[m]).sum() / S[m].sum())
            print(f"  {lo:>3d}-{hi-1:<3d}     {int(m.sum()):>6d}    {pur*100:5.1f}%   "
                  f"{float(blend[m].float().mean())*100:4.1f}%")

    captured = MAJ[big & (MAJ >= 0)]
    uniq, counts = torch.unique(captured, return_counts=True)
    n_cap = int(uniq.numel())
    fragd = int((counts > 1).sum())
    print("---- COVERAGE / FRAGMENTATION (the over-split axis) ----")
    print(f"originals captured (majority of >=1 cluster): {n_cap}/{n_orig} ({n_cap/n_orig*100:.1f}%)")
    print(f"clusters per captured original: median {int(counts.median())}  "
          f"mean {float(counts.float().mean()):.1f}  max {int(counts.max())}")
    print(f"originals split across >1 cluster: {fragd}/{n_cap} ({fragd/max(n_cap,1)*100:.1f}%)")

    return {
        "oracle": name, "ceiling": top1, "size_weighted_purity": frag_in_maj,
        "purity_frac_of_ceiling": frag_in_maj / max(top1, 1e-9),
        "blend_rate": float(blend.float().mean()), "pure_rate": float(pure.float().mean()),
        "originals_captured": n_cap, "originals_total": n_orig,
        "clusters_per_original_mean": float(counts.float().mean()),
        "oversplit_rate": fragd / max(n_cap, 1),
    }, MAJ, blend, pure


def main():
    os.makedirs(QA_OUT, exist_ok=True)
    device = R.get_device()
    print(f"device: {device}")

    base = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True)
    data = torch.from_numpy(base.data).permute(0, 3, 1, 2).float() / 255.0      # [50000,3,32,32]
    targets = np.asarray(base.targets)
    idx0 = client_indices(targets, TARGET_CID)
    cid0_imgs = data[torch.from_numpy(idx0)]                                     # [10000,3,32,32]
    cid0_u8 = base.data[idx0]                                                    # [10000,32,32,3] u8
    labels0 = torch.from_numpy(targets[idx0]).long()
    print(f"client {TARGET_CID}: {len(idx0)} imgs, classes {sorted(set(labels0.tolist()))}")

    gal_ncc  = ncc_feats(cid0_imgs, device).half()
    gal_byol = byol_feats(cid0_imgs, device).half()
    top1_ncc  = calibrate_matcher(cid0_u8, gal_ncc, ncc_feats, device)
    top1_byol = calibrate_matcher(cid0_u8, gal_byol, byol_feats, device)

    f_ncc, f_byol, cl_of_frag, sizes = load_and_embed(device)
    print(f"\n{int(sizes.sum())} fragments in {int((sizes>0).sum())} non-empty clusters "
          f"(median {int(sizes[sizes>0].median())} views/cluster, "
          f"mean {float(sizes[sizes>0].float().mean()):.1f})")

    res_ncc, _, blend_ncc, _ = analyze("NCC (independent, noisy)", f_ncc, gal_ncc,
                                       top1_ncc, labels0, cl_of_frag, sizes, len(idx0))
    res_byol, MAJ, _, pure = analyze("BYOL (strong, clustering-aligned)", f_byol, gal_byol,
                                     top1_byol, labels0, cl_of_frag, sizes, len(idx0))

    # Montages from the BYOL matcher (its instance IDs are trustworthy): a few PURE
    # and a few NCC-flagged BLEND clusters, each as [matched original | its views].
    bins = sorted(glob.glob(os.path.join(CLUSTER_DIR, "bin_*")))
    def montage(mask, tag, k=8):
        sel = torch.where(mask)[0]
        if sel.numel() == 0:
            return
        g = torch.Generator().manual_seed(SEED)
        sel = sel[torch.randperm(sel.numel(), generator=g)[:min(k, sel.numel())]]
        W = 96 * 9                                                        # strip width
        rows = []
        for bi in sel.tolist():
            paths = sorted(glob.glob(os.path.join(bins[bi], "round_*.pt")))[:9]
            fr = torch.stack([to_unit(torch.load(p, map_location="cpu", weights_only=False)) for p in paths])
            strip = Fn.interpolate(make_grid(fr, nrow=9, padding=1)[None], size=(96, W), mode="nearest")[0]
            orig = Fn.interpolate(cid0_imgs[int(MAJ[bi])][None], size=(96, W), mode="nearest")[0]
            rows.append(orig)                                            # top: matched true original
            rows.append(strip)                                          # bottom: this cluster's views
        save_image(make_grid(rows, nrow=1, padding=4), os.path.join(QA_OUT, f"clusters_{tag}.png"))

    montage(pure, "pure")
    montage(blend_ncc, "blend")
    print(f"\nmontages (top: matched original, bottom: cluster views) -> {QA_OUT}/")

    json.dump({"ncc": res_ncc, "byol": res_byol,
               "median_views_per_cluster": int(sizes[sizes > 0].median()),
               "n_clusters": int((sizes >= MIN_CLUSTER).sum()),
               "n_fragments": int(sizes.sum())},
              open(os.path.join(QA_OUT, "purity_summary.json"), "w"), indent=2)
    print(f"summary -> {QA_OUT}/purity_summary.json")


if __name__ == "__main__":
    main()

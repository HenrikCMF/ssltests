"""
regroup_fragments.py -- one-shot pipeline turning the raw, mixed LOKI leak bins
into per-original bins the pretrained reconstructor can invert. Three stages:

  1. SORT   drop blank fragments and the most heavily-cropped ones. Crop severity
            (smooth + low effective resolution = a hard RandomResizedCrop) is scored
            against a CIFAR-100 perturbation bank -- the "unknown reference" proxy,
            so nothing about the real CIFAR-10 originals is assumed.
  2. CLUSTER re-identify each original ACROSS the mixed bins with the BYOL encoder
            (eval_model.pth): embed every usable fragment, build a kNN graph, and
            DBSCAN by cosine density (unknown cluster count). Dense common-class
            blobs are recursively re-split until every cluster is <= 99 fragments
            (the LOKI per-original maximum).
  3. WRITE  lay each cluster out as fragments_clustered/bin_*/round_*.pt.

Then reconstruct with:
    FRAG_DIR=fragments_clustered MIN_VIEWS=4 RECON_SUBDIR=cluster_recons \
        python infer_fragments.py
"""
import glob
import os
import time

import numpy as np
import torch
import torch.nn.functional as Fn
import torchvision

import reconstruction_test as R
from architectures import ResNet18Projv3

# ---------------------------------------------------------------------------- #
# Config
# ---------------------------------------------------------------------------- #
FRAG_DIR     = "fragments/client_00"
OUT_DIR      = "fragments_clustered"
CKPT         = "eval_model.pth"
DATA_DIR     = R.DATA_DIR
BATCH        = 4096
# stage 1 -- crop-severity filter
DROP_CROPPED = False          # False -> keep every non-blank fragment (skip severity)
N_ORIG       = 1000          # CIFAR-100 originals in the calibration bank
N_PERT       = R.NUM_PERTURB  # perturbations per original
SEVERITY_PCT = 0.10          # drop the most-severe (most-cropped) this fraction
BLANK_STD    = 0.05          # per-fragment raw std below this == blank
# stage 2 -- DBSCAN clustering
K_NN         = 32            # neighbours per point in the kNN graph
MIN_PTS      = 4             # min neighbours (incl. self) to be a core point
EPS_SIM      = 0.89          # cosine-sim threshold (where the giant component dissolves)
REFINE_CAP   = 198            # clusters above this are provably multi-original -> re-split
REFINE_EPS   = (0.95, 0.96, 0.97, 0.98)
SEED         = R.SEED

_MEAN = torch.tensor(R.CIFAR_MEAN).view(1, 3, 1, 1)
_STD  = torch.tensor(R.CIFAR_STD).view(1, 3, 1, 1)


def to_unit(x):
    """Per-image min-max to [0,1] -- the canonical fragment->image map (attacks.Loki)."""
    lo = x.amin(dim=(-3, -2, -1), keepdim=True)
    hi = x.amax(dim=(-3, -2, -1), keepdim=True)
    return (x - lo) / (hi - lo + 1e-8)


def _load_round(path):
    """Load one round-major file -> [n, 3, 32, 32] float fragments.

    The extractor (attacks.Loki.save_round_fragments) writes one file per round
    holding all that round's clean fragments as {"frags": [n,3,32,32], ...}.
    A fragment is now identified by (round_file, local_index) instead of its own
    path, since the per-bin directory layout was dropped (clustering re-identifies
    samples across bins from scratch, so the bin index carried no usable signal)."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        obj = obj["frags"]
    return obj.float()


# ---------------------------------------------------------------------------- #
# Stage 1: crop severity
# ---------------------------------------------------------------------------- #
def hf_norm(u):
    """Normalized high-frequency energy |Laplacian(u)|/std. Low == smooth == cropped."""
    lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32,
                       device=u.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    return Fn.conv2d(u, lap, padding=1, groups=3).abs().mean(dim=(1, 2, 3)) / (u.flatten(1).std(1) + 1e-6)


def res_resid(u):
    """Downscale->upscale residual. Small == low effective resolution == upscaled crop."""
    uu = Fn.interpolate(Fn.avg_pool2d(u, 2), scale_factor=2, mode="bilinear", align_corners=False)
    return ((u - uu) ** 2).mean(dim=(1, 2, 3)) / (u.flatten(1).var(1) + 1e-6)


def severity(u, hf_sorted, res_sorted):
    """Crop-severity in [0,2] via the bank ECDFs: high == both smooth AND low-res."""
    n = hf_sorted.numel()
    p_hf = torch.searchsorted(hf_sorted, hf_norm(u).contiguous()).float() / n
    p_res = torch.searchsorted(res_sorted, res_resid(u).contiguous()).float() / n
    return (1.0 - p_hf) + p_res


def calibrate_severity(device):
    """Calibrate the severity ECDFs + drop threshold on a CIFAR-100 perturbation bank."""
    raw = torchvision.datasets.CIFAR100(root=DATA_DIR, train=True, download=True).data
    g = torch.Generator().manual_seed(SEED)
    idx = torch.randperm(raw.shape[0], generator=g).numpy()[:N_ORIG]
    print(f"calibrating crop-severity on {N_ORIG}x{N_PERT} CIFAR-100 perturbations...")
    views = np.empty((N_ORIG * N_PERT, 3, R.IMG_SIZE, R.IMG_SIZE), dtype=np.uint8)
    for i in range(N_ORIG):
        views[i * N_PERT:(i + 1) * N_PERT] = R.pil_views(raw[idx[i]], N_PERT, seed=SEED * 7919 + i)

    hf, res, std = [], [], []
    v = torch.from_numpy(views)
    for i in range(0, v.shape[0], BATCH):
        vb = v[i:i + BATCH].to(device).float()
        u = to_unit(vb)
        hf.append(hf_norm(u).cpu()); res.append(res_resid(u).cpu())
        std.append((vb / 255.0).flatten(1).std(1).cpu())
    hf, res, std = (torch.cat(x) for x in (hf, res, std))
    keep = std >= BLANK_STD
    hf_sorted, res_sorted = hf[keep].sort().values, res[keep].sort().values
    n = hf_sorted.numel()
    p_hf = torch.searchsorted(hf_sorted, hf[keep].contiguous()).float() / n
    p_res = torch.searchsorted(res_sorted, res[keep].contiguous()).float() / n
    thr = float(((1.0 - p_hf) + p_res).quantile(1.0 - SEVERITY_PCT))
    return hf_sorted.to(device), res_sorted.to(device), thr


def sort_usable(hf_sorted, res_sorted, thr, device):
    """Return the usable fragments as (round_file, local_index) items: always drop
    blanks, and -- when DROP_CROPPED -- also drop the most heavily cropped (thr from
    calibrate_severity). thr is None when DROP_CROPPED is off (keep every non-blank).
    Items stay grouped by round file so the downstream loaders hit a 1-file cache."""
    drop_cropped = thr is not None
    files = sorted(glob.glob(os.path.join(FRAG_DIR, "round_*.pt")))
    print(f"sorting {len(files)} round files "
          f"({'drop blank + most-severe %.0f%%' % (SEVERITY_PCT*100) if drop_cropped else 'drop blank only (keep cropped)'})...")
    items = []                                                 # (round_file, local_index)
    t0 = time.time()
    for fi, f in enumerate(files):
        frags = _load_round(f)
        if frags.numel() == 0:
            continue
        live = frags.flatten(1).std(1) >= BLANK_STD            # drop blanks
        live_idx = live.nonzero(as_tuple=True)[0]
        if live_idx.numel():
            if drop_cropped:
                sev = severity(to_unit(frags[live_idx].to(device)), hf_sorted, res_sorted).cpu()
                live_idx = live_idx[sev < thr]
            items += [(f, int(j)) for j in live_idx.tolist()]
        if (fi + 1) % 50 == 0:
            print(f"  {fi+1}/{len(files)} files  ({time.time()-t0:.0f}s)", flush=True)
    print(f"{len(items)} usable fragments")
    return items


# ---------------------------------------------------------------------------- #
# Stage 2: embed + DBSCAN
# ---------------------------------------------------------------------------- #
def embed_all(items, device):
    sd = torch.load(CKPT, map_location="cpu")
    net = ResNet18Projv3()
    net.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("net.")})
    net = net.to(device).eval().requires_grad_(False)
    mean, std = _MEAN.to(device), _STD.to(device)
    print(f"embedding {len(items)} fragments...")
    out = torch.empty(len(items), 512, dtype=torch.float16)
    t0 = time.time()
    cache_path, cache_t = None, None                           # items are file-grouped
    for i in range(0, len(items), BATCH):
        rows = []
        for path, j in items[i:i + BATCH]:
            if path != cache_path:
                cache_t, cache_path = _load_round(path), path
            rows.append(cache_t[j])
        frags = torch.stack(rows).to(device)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            e = Fn.normalize(net.backbone((to_unit(frags) - mean) / std), dim=1)
        out[i:i + e.shape[0]] = e.half().cpu()
    print(f"  embedded in {time.time()-t0:.0f}s")
    return out


def knn_graph(E, k, device):
    Eg = E.to(device); N = Eg.shape[0]
    idx = torch.empty(N, k, dtype=torch.long)
    sim = torch.empty(N, k, dtype=torch.float16)
    for s in range(0, N, BATCH):
        S = Eg[s:s + BATCH] @ Eg.t()
        ar = torch.arange(S.shape[0], device=device)
        S[ar, s + ar] = -2.0                               # mask self
        tk = S.topk(k, dim=1)
        idx[s:s + S.shape[0]] = tk.indices.cpu()
        sim[s:s + S.shape[0]] = tk.values.half().cpu()
    return idx, sim


def dbscan(idx, sim, eps_sim, min_pts, device):
    """DBSCAN over a kNN graph: core = >=min_pts eps-neighbours; density-connected
    cores (min-label propagation) form clusters; border points join an adjacent core;
    everything else is noise (-1). Returns contiguous labels."""
    idx = idx.to(device); sim = sim.to(device)
    N = idx.shape[0]
    nb = sim >= eps_sim
    core = (nb.sum(1) + 1) >= min_pts
    label = torch.arange(N, device=device)
    src, col = torch.where(nb & core[:, None] & core[idx])  # core-core eps edges
    dst = idx[src, col]
    for _ in range(200):
        cand = label.clone()
        cand.scatter_reduce_(0, src, label[dst], reduce="amin")
        cand.scatter_reduce_(0, dst, label[src], reduce="amin")
        new = torch.minimum(label, cand)
        if torch.equal(new, label):
            break
        label = new
    label = torch.where(core, label, torch.full_like(label, -1))
    nonc = ~core                                            # border -> highest-sim core neighbour
    nb_core = nb & core[idx] & nonc[:, None]
    first = torch.where(nb_core, sim, torch.full_like(sim, -2)).argmax(1)
    border = nonc & nb_core.any(1)
    label[border] = label[idx[border, first[border]]]
    lab = label.cpu()
    remap = {int(u): i for i, u in enumerate(torch.unique(lab[lab >= 0]).tolist())}
    return torch.tensor([remap.get(int(x), -1) for x in lab])


def refine(labels, E, device):
    """Re-split every cluster larger than REFINE_CAP at successively tighter eps until
    each is per-original scale; border points dropped during a split become noise."""
    clusters = {}
    for i, c in enumerate(labels.tolist()):
        if c >= 0:
            clusters.setdefault(c, []).append(i)
    final, work = [], [(torch.tensor(m), 0) for m in clusters.values()]
    while work:
        m, ei = work.pop()
        if m.numel() <= REFINE_CAP or ei >= len(REFINE_EPS):
            final.append(m); continue
        Esub = Fn.normalize(E[m].float(), dim=1)               # chunked top-k (see knn_graph):
        idx_sub, sim_sub = knn_graph(Esub, min(K_NN, m.numel() - 1), device)  # avoids a dense
        sub = dbscan(idx_sub, sim_sub, REFINE_EPS[ei], MIN_PTS, device)       # |m|x|m| matrix
        for c in torch.unique(sub):
            if int(c) >= 0:
                work.append((m[sub == c], ei + 1))
    out = torch.full((labels.numel(),), -1, dtype=torch.long)
    for cid, m in enumerate(final):
        out[m] = cid
    return out


# ---------------------------------------------------------------------------- #
# Stage 3: write clusters as bins
# ---------------------------------------------------------------------------- #
def materialize(labels, items):
    clusters = {}
    for i, c in enumerate(labels.tolist()):
        if c >= 0:
            clusters.setdefault(c, []).append(i)
    os.makedirs(OUT_DIR, exist_ok=True)
    order = sorted(clusters.values(), key=lambda m: -len(m))   # bin_00000 = largest
    # Destination per kept global fragment index. The clustered output keeps the
    # per-bin/round layout that infer_fragments expects (written once, not per
    # training round), so only the raw extraction format changed.
    dest = {}
    for bi, members in enumerate(order):
        bdir = os.path.join(OUT_DIR, f"bin_{bi:05d}")
        os.makedirs(bdir, exist_ok=True)
        for ri, gi in enumerate(members):
            dest[gi] = os.path.join(bdir, f"round_{ri:03d}.pt")
    # Group writes by source round file so each round file is loaded exactly once.
    by_file = {}
    for gi, (path, j) in enumerate(items):
        if gi in dest:
            by_file.setdefault(path, []).append((j, gi))
    n_frag = 0
    for path, lst in by_file.items():
        t = _load_round(path)
        for j, gi in lst:
            torch.save(t[j].clone(), dest[gi])
            n_frag += 1
    sizes = sorted((len(m) for m in order), reverse=True)
    print(f"wrote {len(order)} bins / {n_frag} fragments -> {OUT_DIR}/ "
          f"(max {sizes[0]}, median {sizes[len(sizes)//2]})")


def main():
    device = R.get_device()
    print(f"device: {device}")
    if DROP_CROPPED:
        hf_sorted, res_sorted, thr = calibrate_severity(device)
    else:
        hf_sorted = res_sorted = thr = None
    items = sort_usable(hf_sorted, res_sorted, thr, device)
    E = embed_all(items, device)
    idx, sim = knn_graph(E, K_NN, device)
    labels = refine(dbscan(idx, sim, EPS_SIM, MIN_PTS, device), E, device)
    n_clusters = int(labels.max()) + 1
    print(f"{n_clusters} clusters | noise {int((labels < 0).sum())}/{len(items)}")
    materialize(labels, items)


if __name__ == "__main__":
    main()

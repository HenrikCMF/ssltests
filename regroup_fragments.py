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
FRAG_DIR     = os.environ.get("FRAG_DIR", "fragments")
CKPT         = os.environ.get("CKPT", "eval_model.pth")
print(FRAG_DIR, CKPT)
DATA_DIR     = R.DATA_DIR
BATCH        = 512#4096

BLANK_STD    = 0.05          # per-fragment raw std below this == blank
# K_NN caps how many neighbours can link a point, so it is implicitly tuned to the
# expected cluster size -- ~99 rounds x f0 clean fragments per original. A round-
# TRUNCATED arm (the round-cutoff test) has only ~6 rounds, so at K_NN=99 every
# fragment's neighbourhood necessarily spans many originals and DBSCAN merges them
# into a giant component. Env-overridable (default unchanged) so a truncated arm can
# be re-run at a matched K_NN and the cutoff result separated from that artifact.
K_NN         = int(os.environ.get("K_NN", "99"))#32   # neighbours per point in the kNN graph
MIN_PTS      = int(os.environ.get("MIN_PTS", "4"))    # min neighbours (incl. self) to be a core point
EPS_SIM      = float(os.environ.get("EPS_SIM", "0.9"))  # cosine-sim threshold (where the giant component dissolves)
REFINE_CAP   = 99            # clusters above this are provably multi-original -> re-split
REFINE_EPS   = (0.93,0.94,0.95, 0.96, 0.97, 0.98)
SEED         = R.SEED

# --- sign canonicalisation (see _sign_reference) ---------------------------- #
# A fragment is dL/dh_i * x_view / max|.|, and dL/dh_i has no fixed sign under
# BYOL's symmetrised loss, so ~half are saved as photographic negatives. to_unit
# maps -x to 1-to_unit(x) -- a valid-looking image nothing downstream flags -- and
# the BYOL embedding separates a view from its negative, so each original splits
# into a positive and a negative cluster. Canonicalising first is what merges them.
SIGN_CANON     = os.environ.get("SIGN_CANON", "1") == "1"   # 0 = old behaviour (control)
CANON_SEED     = 12345       # attacks.LokiConfig.seed
CANON_N        = 40000       # attacks fc_size = fc_multiplier * local_dataset_size
CANON_MIN_R2   = 0.50        # below this the round is at the DP noise floor -> skip
CANON_DEADBAND = 0.05        # |z - z0| below this: |cutoff| ~ 0, sign unreliable -> skip

# Output keyed to the arm so the canon/control runs never clobber each other; the
# clusters-per-original drop between them is the over-split diagnostic (Remark 6).
OUT_DIR      = os.environ.get(
    "OUT_DIR", "fragments_clustered")

_MEAN = torch.tensor(R.CIFAR_MEAN).view(1, 3, 1, 1)
_STD  = torch.tensor(R.CIFAR_STD).view(1, 3, 1, 1)


def to_unit(x):
    """Per-image min-max to [0,1] -- the canonical fragment->image map (attacks.Loki)."""
    lo = x.amin(dim=(-3, -2, -1), keepdim=True)
    hi = x.amax(dim=(-3, -2, -1), keepdim=True)
    return (x - lo) / (hi - lo + 1e-8)


_Z_ORDER = None


def _z_order():
    """Order statistics of the fixed standard-normal draw behind every round's cutoffs.

    attacks.setup_fc_biases re-seeds its generator from cfg.seed on *every* call, so
    the draw is identical in every round; only (bias_mean, bias_std) drift as the
    server refines the distribution. Sorting commutes with a positive affine map, so
        cutoff_i = bias_mean + bias_std * z_(i)
    and sign(cutoff_i) therefore flips at the single index where z_(i) = -bias_mean/bias_std."""
    global _Z_ORDER
    if _Z_ORDER is None:
        g = torch.Generator().manual_seed(CANON_SEED)
        _Z_ORDER = torch.sort(torch.randn(CANON_N, generator=g))[0].numpy()
    return _Z_ORDER


def _fit_crossing(zz, y, nwin=120, ncand=600):
    """Recover z0 = -bias_mean/bias_std for one round from that round's own fragments.

    A bin fires only for views whose mean brightness lands in its band, so
    |fragment mean| ~ |cutoff_bin| / max|x|: binned medians of |fragment mean| against
    z_(bin) trace a V whose vertex sits at z0. Fit y = a|z - z0| + b over candidate
    vertices and keep the best. The returned R^2 collapses once a round reaches the DP
    noise floor -- which is precisely when its fragments carry no signal to recover."""
    o = np.argsort(zz)
    zz, y = zz[o], y[o]
    W = len(zz) // nwin
    if W < 8:
        return 0.0, 0.0
    zc = np.array([zz[i * W:(i + 1) * W].mean() for i in range(nwin)])
    yc = np.array([np.median(y[i * W:(i + 1) * W]) for i in range(nwin)])
    cand = np.linspace(zc.min(), zc.max(), ncand)
    X = np.abs(zc[None, :] - cand[:, None])                    # [ncand, nwin]
    xm, ym = X.mean(1, keepdims=True), yc.mean()
    xv = X - xm
    a = (xv * (yc - ym)).sum(1) / np.maximum((xv * xv).sum(1), 1e-12)
    b = ym - a * xm[:, 0]
    sse = ((a[:, None] * X + b[:, None] - yc) ** 2).sum(1)
    k = int(np.argmin(sse))
    sst = float(((yc - ym) ** 2).sum())
    return float(cand[k]), float(1.0 - sse[k] / max(sst, 1e-12))


def _sign_reference(obj, mu):
    """Sign each fragment's mean *should* carry: +1/-1, or 0 where it is unreliable.

    Prefers the exact per-fragment cutoffs recorded by the attack; falls back to
    fitting the crossing for runs extracted before those were saved. Returns
    (want, note), with want=None when the round cannot be canonicalised at all."""
    c = obj.get("cutoffs")
    if c is not None and tuple(c.shape) == mu.shape:
        c = c.float().numpy()
        dead = CANON_DEADBAND * (float(c.std()) + 1e-12)       # z-units * bias_std
        want = np.where(np.abs(c) < dead, 0.0, np.sign(c))
        return want, f"exact cutoffs, {int((want == 0).sum())} in deadband"
    bins = obj.get("bins")
    if bins is None:
        return None, "no bins recorded"
    z = _z_order()
    b = bins.numpy()
    if int(b.max()) >= len(z):
        return None, f"bin {int(b.max())} >= CANON_N={len(z)}"
    zz = z[b]
    z0, r2 = _fit_crossing(zz, np.abs(mu))
    if r2 < CANON_MIN_R2:
        return None, f"R2={r2:.3f} < {CANON_MIN_R2}, at noise floor -- left as-is"
    d = zz - z0
    want = np.where(np.abs(d) < CANON_DEADBAND, 0.0, np.sign(d))
    return want, f"fitted z0={z0:+.4f} R2={r2:.3f}, {int((want == 0).sum())} in deadband"


_CANON = {}                                  # path -> (want, note); fitted once per file
_CANON_N = [0, 0]                            # [flipped, total] across all files


def _canonicalise(path, obj, frags):
    """Flip fragments saved as negatives so every view uses the +x convention.

    Per-fragment and within one round only -- the bin index is used as that round's
    brightness readout, never as a cross-round identity key (views drift across bins
    every round, which is why clustering re-identifies originals by embedding). The
    target is a *global* convention, so two views of one original from different
    rounds and different bins both come out as +x and embed together."""
    mu = frags.flatten(1).mean(1).numpy()
    if path not in _CANON:
        _CANON[path] = _sign_reference(obj, mu)
        print(f"  canon {os.path.basename(path)}: {_CANON[path][1]}")
    want, _ = _CANON[path]
    if want is None:
        return frags
    flip = torch.from_numpy((want * np.sign(mu)) < 0)
    frags[flip] = -frags[flip]
    _CANON_N[0] += int(flip.sum())
    _CANON_N[1] += flip.numel()
    return frags


def _load_round(path):
    """Load one round-major file -> [n, 3, 32, 32] float fragments.

    The extractor (attacks.Loki.save_round_fragments) writes one file per round
    holding all that round's clean fragments as {"frags": [n,3,32,32], ...}.
    A fragment is now identified by (round_file, local_index) instead of its own
    path, since the per-bin directory layout was dropped (clustering re-identifies
    samples across bins from scratch, so the bin index carried no usable signal).

    Fragments are sign-canonicalised here (SIGN_CANON=0 disables) so that the single
    change covers both the embedding in `embed_all` and the fragments `materialize`
    writes out -- everything downstream reads OUT_DIR, not the raw extraction."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        return obj.float()
    frags = obj["frags"].float()
    if SIGN_CANON and frags.numel():
        frags = _canonicalise(path, obj, frags)
    return frags

def collect_usable():
    """Return non-blank fragments as (round_file, local_index) items, grouped by
    file so the downstream loaders hit a 1-file cache."""
    files = sorted(glob.glob(os.path.join(FRAG_DIR, "round_*.pt")))
    print(f"scanning {len(files)} round files...")
    items = []
    t0 = time.time()
    for fi, f in enumerate(files):
        frags = _load_round(f)
        if frags.numel() == 0:
            continue
        live_idx = (frags.flatten(1).std(1) >= BLANK_STD).nonzero(as_tuple=True)[0]
        if live_idx.numel():
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
    cache_path, cache_t = None, None                           # items are file-grouped
    for i in range(0, len(items), BATCH):
        rows = []
        for path, j in items[i:i + BATCH]:
            if path != cache_path:
                cache_t, cache_path = _load_round(path), path
            rows.append(cache_t[j])
        frags = torch.stack(rows).to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            e = Fn.normalize(net.backbone((to_unit(frags) - mean) / std), dim=1)
        out[i:i + e.shape[0]] = e.half().cpu()
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


def canon_summary():
    """One-pass flip statistics. Call right after collect_usable -- the later stages
    re-load the same files (the fit is cached, but the counters would double-count)."""
    if not SIGN_CANON:
        print("sign canonicalisation OFF (control run)")
        return
    ok = sum(1 for w, _ in _CANON.values() if w is not None)
    ex = sum(1 for w, n in _CANON.values() if w is not None and n.startswith("exact"))
    fl, tot = _CANON_N
    print(f"sign canon: {ok}/{len(_CANON)} rounds usable ({ex} exact, {ok - ex} fitted) | "
          f"{fl}/{tot} fragments flipped ({100 * fl / max(tot, 1):.1f}%)")
    _CANON_N[0] = _CANON_N[1] = 0


def main():
    device = R.get_device()
    print(f"device: {device}")
    items = collect_usable()
    canon_summary()
    E = embed_all(items, device)
    idx, sim = knn_graph(E, K_NN, device)
    labels = refine(dbscan(idx, sim, EPS_SIM, MIN_PTS, device), E, device)
    n_clusters = int(labels.max()) + 1
    print(f"{n_clusters} clusters | noise {int((labels < 0).sum())}/{len(items)}")
    materialize(labels, items)


if __name__ == "__main__":
    main()

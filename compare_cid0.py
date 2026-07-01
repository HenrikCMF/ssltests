"""
compare_cid0.py -- validate the cluster reconstructions against the REAL data that
the LOKI target (client 0) actually held.

We never used client 0's images anywhere in the pipeline, so they are an honest
ground truth. Client 0's partition is fully determined by the run config
(NUM_CLIENTS=5, classes_per_client=2, seed=12345 -> 2 full CIFAR-10 classes,
10k images). For every reconstruction in reconstruction_out/cluster_recons we:

  * find its nearest CIFAR-10 train image over the FULL 50k set (mean-subtracted
    normalized cross-correlation on lightly-blurred RGB -- robust to the recon's
    colour drift / blur), and check whether that match falls inside client 0's set.
    Chance is 10k/50k = 20%; a faithful leak lands far above it.
  * compare best-match scores against client 0 vs a non-target client (client 1,
    different classes) -- the null. Real leaks score much higher against client 0.
  * dump upscaled [recon | best client-0 match] montages for eyeballing.
"""
import glob
import os

import lpips
import numpy as np
import torch
import torch.nn.functional as Fn
import torchvision
from PIL import Image
from torchvision.utils import make_grid, save_image

import reconstruction_test as R

RECON_DIR   = "reconstruction_out/cluster_recons"
CLUSTER_DIR = "fragments_clustered"        # bins that produced the reconstructions
DATA_DIR    = R.DATA_DIR
NUM_CLIENTS = 5
CLS_PER     = 2
SEED        = 12345
TARGET_CID  = 0
NULL_CID    = 1
QA_OUT      = "cid0_compare"
LPIPS_K     = 8       # NCC candidates re-ranked by LPIPS to pick the best match
LP_GATE     = 0.23    # LPIPS gate for the boundary montage (lower = more similar)
# precision-weighted leak estimate: no single LPIPS gate works (genuine leaks exist
# at every LPIPS level, just sparser), so bin each image's best LPIPS and weight by
# the eyeballed genuine fraction g per bin (read off the match_lpbin_* montages).
LP_BINS     = (0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 1.0)
LP_GENUINE  = (1.00, 0.90, 0.85, 0.55, 0.30, 0.12, 0.03)   # g per bin -- eyeballed from match_lpbin_*


def client_indices(targets, cid):
    """Replicates dataloader.CIFAR10BYOLClientData._partition_indices (the active
    shard method) to recover the exact train indices held by `cid`."""
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


def ncc_feats(imgs, device, blur=2):
    """imgs [N,3,32,32] in [0,1] -> mean-subtracted, L2-normalized, blurred feature
    rows for cross-correlation matching (robust to brightness/contrast and blur)."""
    x = Fn.avg_pool2d(imgs.to(device), blur)            # light blur: 32->16
    x = x.flatten(1)
    x = x - x.mean(1, keepdim=True)
    return Fn.normalize(x, dim=1)


def ssim_pairs(a, b, device, ws=11, sigma=1.5, chunk=4096):
    """Per-pair windowed SSIM for a,b [N,3,32,32] in [0,1] -> [N]."""
    c = torch.arange(ws, device=device).float() - (ws - 1) / 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    w = (g[:, None] @ g[None, :]).expand(3, 1, ws, ws).contiguous()
    pad, c1, c2 = ws // 2, 0.01 ** 2, 0.03 ** 2
    out = torch.empty(a.shape[0], device=device)
    for s in range(0, a.shape[0], chunk):
        ai, bi = a[s:s + chunk].to(device), b[s:s + chunk].to(device)
        mu_a = Fn.conv2d(ai, w, padding=pad, groups=3); mu_b = Fn.conv2d(bi, w, padding=pad, groups=3)
        a2 = Fn.conv2d(ai * ai, w, padding=pad, groups=3) - mu_a ** 2
        b2 = Fn.conv2d(bi * bi, w, padding=pad, groups=3) - mu_b ** 2
        ab = Fn.conv2d(ai * bi, w, padding=pad, groups=3) - mu_a * mu_b
        sij = ((2 * mu_a * mu_b + c1) * (2 * ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (a2 + b2 + c2))
        out[s:s + ai.shape[0]] = sij.mean((1, 2, 3))
    return out


def best_match(recon_f, gt_f, chunk=2048):
    """For each recon row, best (score, gt index) over gt_f by dot product."""
    idx = torch.empty(recon_f.shape[0], dtype=torch.long, device=recon_f.device)
    sc = torch.empty(recon_f.shape[0], device=recon_f.device)
    for s in range(0, recon_f.shape[0], chunk):
        m = recon_f[s:s + chunk] @ gt_f.t()
        mx = m.max(1)
        idx[s:s + chunk] = mx.indices; sc[s:s + chunk] = mx.values
    return sc, idx


def unique_best_mask(scores, idx):
    """Boolean keep-mask over recons enforcing one recon per matched gt image:
    walk recons from highest to lowest score and keep a recon only if its matched
    index hasn't been claimed yet. Lower-scoring recons landing on an already-
    claimed image are dropped, so each image is counted by its best recon only."""
    keep = torch.zeros_like(scores, dtype=torch.bool)
    seen = set()
    for r in scores.argsort(descending=True).tolist():
        g = int(idx[r])
        if g not in seen:
            seen.add(g)
            keep[r] = True
    return keep


def topk_ncc(rf, gt_f, k, chunk=2048):
    """Top-k candidate gt indices per recon by NCC dot product -> [N,k] (cpu longs)."""
    out = torch.empty(rf.shape[0], k, dtype=torch.long)
    for s in range(0, rf.shape[0], chunk):
        out[s:s + chunk] = (rf[s:s + chunk] @ gt_f.t()).topk(k, dim=1).indices.cpu()
    return out


def lpips_match(recons, gt_imgs, cand_idx, model, device, sz=64, chunk=256):
    """Re-rank each recon's candidate gt images by LPIPS (perceptual distance) and
    return (best_dist [N], best_gt_index [N]): the perceptually closest of the
    candidates. SSIM is harsh on the recons' blur/colour drift, so LPIPS recovers
    faithful leaks SSIM misses. Inputs in [0,1]; upsampled to sz so AlexNet
    features aren't degenerate at 32px. best_gt_index is a position in gt_imgs."""
    n, k = cand_idx.shape
    dist = torch.empty(n); idx = torch.empty(n, dtype=torch.long)
    up = lambda im: Fn.interpolate(im, size=sz, mode="bilinear", align_corners=False)
    for s in range(0, n, chunk):
        ci = cand_idx[s:s + chunk]; b = ci.shape[0]
        r = up(recons[s:s + chunk].to(device))[:, None].expand(b, k, 3, sz, sz)
        g = up(gt_imgs[ci.reshape(-1)].to(device))
        with torch.no_grad():
            d = model(r.reshape(b * k, 3, sz, sz), g, normalize=True).reshape(b, k)
        dmin = d.min(1)
        dist[s:s + b] = dmin.values.cpu()
        idx[s:s + b] = ci.gather(1, dmin.indices.cpu()[:, None]).squeeze(1)
    return dist, idx


def main():
    device = R.get_device()
    base = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True)
    data = torch.from_numpy(base.data).permute(0, 3, 1, 2).float() / 255.0   # [50000,3,32,32]
    targets = np.asarray(base.targets)
    idx0 = client_indices(targets, TARGET_CID)
    idx1 = client_indices(targets, NULL_CID)
    cls0 = sorted(set(targets[idx0].tolist()))
    print(f"client {TARGET_CID}: {len(idx0)} imgs, classes {cls0} | "
          f"null client {NULL_CID}: classes {sorted(set(targets[idx1].tolist()))}")

    files = sorted(glob.glob(os.path.join(RECON_DIR, "*.png")))
    recons = torch.stack([torch.from_numpy(np.asarray(Image.open(f), dtype=np.uint8))
                          .permute(2, 0, 1).float() / 255.0 for f in files])
    print(f"{len(recons)} reconstructions from {RECON_DIR}")

    rf = ncc_feats(recons, device)
    full_f = ncc_feats(data, device)
    gt0_f = full_f[torch.from_numpy(idx0).to(device)]
    gt1_f = full_f[torch.from_numpy(idx1).to(device)]

    # 1) full-set match: does the nearest of ALL 50k land in client 0's set?
    sc_full, idx_full = best_match(rf, full_f)

    # enforce one match per image: if several recons share the same nearest
    # CIFAR-10 image, keep only the best-scoring one so a single leaked picture
    # isn't double-counted (e.g. several clusters reconstructing the same image).
    keep = unique_best_mask(sc_full, idx_full); kc = keep.cpu()
    n_before, n_keep = len(recons), int(keep.sum())
    files  = [f for f, k in zip(files, kc.tolist()) if k]
    recons = recons[kc]
    rf, sc_full, idx_full = rf[keep], sc_full[keep], idx_full[keep]
    print(f"\nunique-match filter: kept {n_keep}/{n_before} recons "
          f"({n_before - n_keep} dropped as duplicates of an already-matched image)")

    in0 = torch.isin(idx_full, torch.from_numpy(idx0).to(device))
    lab_full = torch.from_numpy(targets).to(device)[idx_full]
    in_cls0 = torch.isin(lab_full, torch.tensor(cls0, device=device))
    print(f"\nfull-set nearest match in client-0 SET : {float(in0.float().mean())*100:5.1f}%  "
          f"(chance {len(idx0)/len(data)*100:.0f}%)")
    print(f"full-set nearest match in client-0 CLASSES: {float(in_cls0.float().mean())*100:5.1f}%  "
          f"(chance {len(idx0)/len(data)*100:.0f}%)")
    # distinct client-0 images leaked: unique gt indices the recons land on that
    # belong to client 0 (<= len(idx0)=10k; dedup already makes idx_full unique).
    n_leak = int(torch.unique(idx_full[in0]).numel())
    print(f"distinct client-0 images leaked          : {n_leak}/{len(idx0)}  "
          f"({n_leak/len(idx0)*100:.1f}% of client 0)")

    # 2) null test: best score against client 0 vs the non-target client
    sc0, idx0_match = best_match(rf, gt0_f)
    sc1, _ = best_match(rf, gt1_f)
    print(f"\nbest NCC vs client 0  : median {float(sc0.median()):.3f}  p90 {float(sc0.quantile(.9)):.3f}")
    print(f"best NCC vs client 1  : median {float(sc1.median()):.3f}  p90 {float(sc1.quantile(.9)):.3f}  (null)")
    print(f"recons matching client 0 better than client 1: {float((sc0 > sc1).float().mean())*100:.1f}%")

    # 3) SSIM of each recon vs its best client-0 match (structural fidelity)
    gt0_imgs = data[torch.from_numpy(idx0)]
    ss0 = ssim_pairs(recons, gt0_imgs[idx0_match.cpu()], device)
    n = ss0.numel()
    n5, n7 = int((ss0 > 0.5).sum()), int((ss0 > 0.7).sum())
    print(f"\nSSIM(recon, client-0 match): median {float(ss0.median()):.3f}  "
          f"p10 {float(ss0.quantile(.1)):.3f}  p90 {float(ss0.quantile(.9)):.3f}")
    print(f"  > 0.5: {n5}/{n} ({n5/n*100:.1f}%)   > 0.7: {n7}/{n} ({n7/n*100:.1f}%)")
    # distinct client-0 images actually reconstructed: an image counts as leaked
    # only if >=1 recon clears the SSIM bar against it. This is the defensible
    # leak count -- it gates on fidelity, not just nearest-neighbour proximity.
    gleak = torch.from_numpy(idx0).to(device)[idx0_match]   # global CIFAR idx of each recon's client-0 match
    d5 = int(torch.unique(gleak[ss0 > 0.5]).numel())
    d7 = int(torch.unique(gleak[ss0 > 0.7]).numel())
    print(f"  distinct client-0 leaked @SSIM>0.5: {d5}/{len(idx0)} ({d5/len(idx0)*100:.1f}%)   "
          f"@SSIM>0.7: {d7}/{len(idx0)} ({d7/len(idx0)*100:.1f}%)")

    # 3c) LPIPS re-match: SSIM under-credits the recons' blur/colour drift, so
    # re-rank each recon's top-K NCC client-0 candidates by LPIPS (perceptual) and
    # gate on that. lp_j is a position in client-0's set; lp is the distance (lower
    # = more similar). Lets us catch faithful leaks that sit below the SSIM bar.
    lp_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for p in lp_model.parameters():
        p.requires_grad_(False)
    cand = topk_ncc(rf, gt0_f, LPIPS_K)
    lp, lp_j = lpips_match(recons, gt0_imgs, cand, lp_model, device)
    print(f"\nLPIPS(recon, best client-0 match): median {float(lp.median()):.3f}  "
          f"p10 {float(lp.quantile(.1)):.3f}  p90 {float(lp.quantile(.9)):.3f}  (lower = better)")
    gleak_lp = torch.from_numpy(idx0)[lp_j]   # global CIFAR idx of each recon's LPIPS match
    print("  distinct client-0 leaked @LPIPS<t:")
    for t in (0.10, 0.15, 0.20, 0.25, 0.30):
        d = int(torch.unique(gleak_lp[lp < t]).numel())
        print(f"     <{t:.2f}: {d:>5d}/{len(idx0)} ({d/len(idx0)*100:4.1f}%)")

    # 3d) precision-weighted leak estimate: take each leaked image's BEST recon,
    # bin by its LPIPS, and weight each bin by its eyeballed genuine fraction. This
    # credits the sparse genuine leaks in high-LPIPS bins instead of discarding them
    # at a hard gate. Sum Sum(N_bin * g_bin) -> expected distinct images leaked.
    keep_img = unique_best_mask(-lp, lp_j)          # one recon per client-0 image (its best)
    img_r = keep_img.nonzero(as_tuple=True)[0]      # recon idx, one per matched image
    img_lp = lp[img_r]                              # that image's best LPIPS
    print(f"\nprecision-weighted leak estimate ({img_r.numel()} of {len(idx0)} images matched):")
    print("  LPIPS bin     images   g(eyeball)   est. genuine")
    est = 0.0
    for (lo, hi), g in zip(zip(LP_BINS[:-1], LP_BINS[1:]), LP_GENUINE):
        nb = int(((img_lp >= lo) & (img_lp < hi)).sum()); est += nb * g
        print(f"  {lo:.2f}-{hi:.2f}    {nb:>6d}     {g:.2f}        {nb*g:7.1f}")
    print(f"  TOTAL distinct client-0 leaked (precision-weighted): "
          f"{est:.0f}/{len(idx0)} ({est/len(idx0)*100:.1f}%)")

    # 3b) does cluster size (number of fragments) predict reconstruction SSIM?
    sizes = torch.tensor([len(glob.glob(os.path.join(
        CLUSTER_DIR, os.path.splitext(os.path.basename(f))[0], "round_*.pt"))) for f in files]).float()
    s = ss0.cpu()
    def corr(a, b):
        a, b = a - a.mean(), b - b.mean()
        return float((a * b).sum() / (a.norm() * b.norm() + 1e-9))
    spearman = corr(sizes.argsort().argsort().float(), s.argsort().argsort().float())
    print(f"\ncluster size vs SSIM:  Pearson {corr(sizes, s):+.3f}   Spearman {spearman:+.3f}")
    print("  size bucket    n      meanSSIM   >0.7")
    edges = [2, 4, 6, 8, 12, 20, 35, 100]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (sizes >= lo) & (sizes < hi)
        if bool(m.any()):
            sm = s[m]
            print(f"  {lo:>3d}-{hi-1:<3d}    {int(m.sum()):>6d}    {float(sm.mean()):.3f}     "
                  f"{float((sm > 0.7).float().mean())*100:4.1f}%")

    # 4) montages: best / median / worst recons beside their client-0 match
    os.makedirs(QA_OUT, exist_ok=True)
    order = sc0.argsort(descending=True).cpu()
    # recons sitting right at the SSIM>0.7 gate: the marginal pairs that decide the
    # distinct-leaked count. Sorted just-above -> just-below the line so you can see
    # whether 0.70 is "same specific image" or only "same class".
    near07 = (ss0 - 0.7).abs().argsort()[:32]
    near07 = near07[ss0[near07].argsort(descending=True)].cpu()
    for tag, rows in [("best", order[:32]), ("median", order[len(order)//2 - 16:len(order)//2 + 16]),
                      ("worst", order[-32:]), ("at_ssim0.70", near07)]:
        pairs = []
        for r in rows.tolist():
            up = lambda im: Fn.interpolate(im[None], scale_factor=4, mode="nearest")[0]
            pairs.append(up(recons[r])); pairs.append(up(gt0_imgs[idx0_match[r].cpu()]))
        grid = make_grid(torch.stack(pairs), nrow=8, padding=2)   # recon,gt,recon,gt,...
        save_image(grid, os.path.join(QA_OUT, f"match_{tag}.png"))

    # LPIPS montages: best / boundary / worst by perceptual distance (lower=better).
    # The boundary set decides distinct-leaked @LPIPS<LP_GATE, same as at_ssim0.70.
    lp_order = lp.argsort()                       # ascending: best (lowest LPIPS) first
    near_lp = (lp - LP_GATE).abs().argsort()[:32]
    near_lp = near_lp[lp[near_lp].argsort()]      # sorted across the gate, best->worst
    for tag, rows in [("lpips_best", lp_order[:32]), (f"lpips_at_{LP_GATE:.2f}", near_lp),
                      ("lpips_worst", lp_order[-32:])]:
        pairs = []
        for r in rows.tolist():
            up = lambda im: Fn.interpolate(im[None], scale_factor=4, mode="nearest")[0]
            pairs.append(up(recons[r])); pairs.append(up(gt0_imgs[lp_j[r]]))
        grid = make_grid(torch.stack(pairs), nrow=8, padding=2)
        save_image(grid, os.path.join(QA_OUT, f"match_{tag}.png"))

    # per-LPIPS-bin montages: evenly sampled across each bin so you can read off the
    # genuine fraction g that feeds the precision-weighted estimate (LP_GENUINE).
    for lo, hi in zip(LP_BINS[:-1], LP_BINS[1:]):
        m = ((img_lp >= lo) & (img_lp < hi)).nonzero(as_tuple=True)[0]
        if m.numel() == 0:
            continue
        sel = img_r[m[torch.linspace(0, m.numel() - 1, min(32, m.numel())).long()]]
        pairs = []
        for r in sel.tolist():
            up = lambda im: Fn.interpolate(im[None], scale_factor=4, mode="nearest")[0]
            pairs.append(up(recons[r])); pairs.append(up(gt0_imgs[lp_j[r]]))
        grid = make_grid(torch.stack(pairs), nrow=8, padding=2)
        save_image(grid, os.path.join(QA_OUT, f"match_lpbin_{lo:.2f}-{hi:.2f}.png"))
    print(f"\nmontages (recon | client-0 match) -> {QA_OUT}/  [pairs are adjacent columns]")


if __name__ == "__main__":
    main()

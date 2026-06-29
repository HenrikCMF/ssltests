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
    in0 = torch.isin(idx_full, torch.from_numpy(idx0).to(device))
    lab_full = torch.from_numpy(targets).to(device)[idx_full]
    in_cls0 = torch.isin(lab_full, torch.tensor(cls0, device=device))
    print(f"\nfull-set nearest match in client-0 SET : {float(in0.float().mean())*100:5.1f}%  "
          f"(chance {len(idx0)/len(data)*100:.0f}%)")
    print(f"full-set nearest match in client-0 CLASSES: {float(in_cls0.float().mean())*100:5.1f}%  "
          f"(chance {len(idx0)/len(data)*100:.0f}%)")

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
    for tag, rows in [("best", order[:32]), ("median", order[len(order)//2 - 16:len(order)//2 + 16]),
                      ("worst", order[-32:])]:
        pairs = []
        for r in rows.tolist():
            up = lambda im: Fn.interpolate(im[None], scale_factor=4, mode="nearest")[0]
            pairs.append(up(recons[r])); pairs.append(up(gt0_imgs[idx0_match[r].cpu()]))
        grid = make_grid(torch.stack(pairs), nrow=8, padding=2)   # recon,gt,recon,gt,...
        save_image(grid, os.path.join(QA_OUT, f"match_{tag}.png"))
    print(f"\nmontages (recon | client-0 match) -> {QA_OUT}/  [pairs are adjacent columns]")


if __name__ == "__main__":
    main()

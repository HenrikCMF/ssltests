"""
score_recons.py -- score one or more reconstruction directories under ONE fixed
criterion, and emit machine-readable JSON.

Why this exists. Table 3 of privacy_model.pdf pits supply/m* = 0.11/0.34/1.09/10.9/17.4
against an "observed" column reading 0%/0%/6%/substantial/full. That column mixes success
criteria between rows and two of its entries could not be traced to any artifact in the
repo, which makes the paper's headline evidence unfalsifiable for a reader. This script
re-scores every arm with the SAME gate, the same matcher and the same ground truth, so the
observed column becomes one number per eps computed one way.

It is deliberately a thin wrapper: every metric is computed by the functions in
compare_cid0.py (imported, not reimplemented), so the numbers are the same pipeline that
produced the existing end-to-end figures. What it adds is (a) the LPIPS < 0.23 gate the
model actually quotes -- compare_cid0's print loop hardcodes 0.10..0.30 and skips 0.23 --
(b) JSON output, and (c) one process for N arms so the CIFAR pool, the NCC features and
the vgg-LPIPS network are built once instead of N times.

Usage:
    python score_recons.py out.json  name1=recon_dir1  name2=recon_dir2 ...
"""
import glob
import json
import os
import sys

import lpips
import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image

import compare_cid0 as C
import reconstruction_test as R

# Every gate reported for every arm. The two the model quotes are SSIM > 0.7 and
# vgg-LPIPS < 0.23 (privacy_model.pdf Table 2 / correction ledger); the rest are
# reported so a reader can see the ranking is not an artifact of one gate.
SSIM_GATES  = (0.5, 0.6, 0.7)
LPIPS_GATES = (0.15, 0.20, 0.23, 0.25, 0.30)


def score_one(files, recons, data, idx0, idx1, full_f, gt0_f, gt1_f, gt0_imgs,
              targets, lp_model, device):
    rf = C.ncc_feats(recons, device)

    # 1) nearest over the whole participating pool, deduped so one leaked image is
    # never counted by several clusters that reconstructed it.
    sc_full, idx_full = C.best_match(rf, full_f)
    keep = C.unique_best_mask(sc_full, idx_full)
    n_before, n_keep = len(recons), int(keep.sum())
    recons = recons[keep.cpu()]
    rf, sc_full, idx_full = rf[keep], sc_full[keep], idx_full[keep]

    in0 = torch.isin(idx_full, torch.from_numpy(idx0).to(device))
    n_nearest = int(torch.unique(idx_full[in0]).numel())

    # 2) null test: does a recon match the target client better than a non-target one?
    sc0, idx0_match = C.best_match(rf, gt0_f)
    sc1, _ = C.best_match(rf, gt1_f)

    # 3) SSIM against the best client-0 match
    ss0 = C.ssim_pairs(recons, gt0_imgs[idx0_match.cpu()], device)
    gleak = torch.from_numpy(idx0).to(device)[idx0_match]

    # 4) LPIPS re-rank of the top-K NCC candidates (SSIM is harsh on the recons' blur)
    cand = C.topk_ncc(rf, gt0_f, C.LPIPS_K)
    lp, lp_j = C.lpips_match(recons, gt0_imgs, cand, lp_model, device)
    gleak_lp = torch.from_numpy(idx0)[lp_j]

    n0 = len(idx0)
    out = {
        "n_bins_reconstructed": n_before,
        "n_recons_after_dedup": n_keep,
        "nearest_in_client0_pct": float(in0.float().mean()) * 100,
        "null_beats_client1_pct": float((sc0 > sc1).float().mean()) * 100,
        "ssim_median": float(ss0.median()),
        "lpips_median": float(lp.median()),
        "distinct_leaked_nearest": n_nearest,
        "distinct_leaked_nearest_pct": n_nearest / n0 * 100,
        "client0_size": n0,
    }
    for g in SSIM_GATES:
        d = int(torch.unique(gleak[ss0 > g]).numel())
        out[f"distinct_ssim_gt_{g}"] = d
        out[f"distinct_ssim_gt_{g}_pct"] = d / n0 * 100
    for g in LPIPS_GATES:
        d = int(torch.unique(gleak_lp[lp < g]).numel())
        out[f"distinct_lpips_lt_{g}"] = d
        out[f"distinct_lpips_lt_{g}_pct"] = d / n0 * 100
    return out


def main():
    out_path, arms = sys.argv[1], []
    for a in sys.argv[2:]:
        name, _, d = a.partition("=")
        arms.append((name, d))

    device = R.get_device()
    print(f"device: {device}  |  {len(arms)} arms", flush=True)

    data, targets = C.load_dataset()
    idx0, idx1 = C.client_indices(targets, C.TARGET_CID), C.client_indices(targets, C.NULL_CID)
    print(f"pool {len(data)} | client0 {len(idx0)} | client1 {len(idx1)}", flush=True)

    full_f = C.ncc_feats(data, device)
    gt0_f = full_f[torch.from_numpy(idx0).to(device)]
    gt1_f = full_f[torch.from_numpy(idx1).to(device)]
    gt0_imgs = data[torch.from_numpy(idx0)]

    lp_model = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
    for p in lp_model.parameters():
        p.requires_grad_(False)

    results = {}
    if os.path.exists(out_path):
        results = json.load(open(out_path))          # resumable across invocations
    for name, d in arms:
        files = sorted(glob.glob(os.path.join(d, "*.png")))
        if not files:
            print(f"[{name}] NO RECONS in {d} -- recording zero", flush=True)
            results[name] = {"recon_dir": d, "n_bins_reconstructed": 0,
                             "n_recons_after_dedup": 0, "client0_size": len(idx0),
                             "distinct_leaked_nearest": 0, "distinct_leaked_nearest_pct": 0.0,
                             **{f"distinct_ssim_gt_{g}": 0 for g in SSIM_GATES},
                             **{f"distinct_ssim_gt_{g}_pct": 0.0 for g in SSIM_GATES},
                             **{f"distinct_lpips_lt_{g}": 0 for g in LPIPS_GATES},
                             **{f"distinct_lpips_lt_{g}_pct": 0.0 for g in LPIPS_GATES}}
            json.dump(results, open(out_path, "w"), indent=2)
            continue
        recons = torch.stack([torch.from_numpy(np.asarray(Image.open(f), dtype=np.uint8))
                              .permute(2, 0, 1).float() / 255.0 for f in files])
        print(f"[{name}] {len(recons)} recons from {d}", flush=True)
        r = score_one(files, recons, data, idx0, idx1, full_f, gt0_f, gt1_f, gt0_imgs,
                      targets, lp_model, device)
        r["recon_dir"] = d
        results[name] = r
        print(f"[{name}] distinct leaked: nearest {r['distinct_leaked_nearest_pct']:.1f}%  "
              f"@SSIM>0.7 {r['distinct_ssim_gt_0.7_pct']:.1f}%  "
              f"@LPIPS<0.23 {r['distinct_lpips_lt_0.23_pct']:.1f}%", flush=True)
        json.dump(results, open(out_path, "w"), indent=2)

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

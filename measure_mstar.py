"""Measure m* directly: reconstruction quality vs number of clean fragments.

Replaces the derived m* of the privacy model (van Trees + J_aug + a Gaussian
prior) with the thing it was trying to predict: run the *actual* inverter on
m in {1,2,4,...,64} clean views of one original at sigma_rec = 0 and read off
how many views it needs to clear a given quality bar.

Deleting the derivation also deletes its four weak points -- J_aug, the Gaussian
image prior, the "A_i random of KNOWN form" problem, and the non-monotonicity of
MSE under the pipeline's per-image min-max. van Trees is demoted to what it is
actually good for: bounding how much headroom a better attacker has over ours.

Inputs are built to match the real fragment space exactly:
  * originals  = client 0's own CIFAR-10 shard (the data the attack targets)
  * views      = reconstruction_test.PRECOMPUTE_TF, the bit-exact client augmentation
  * input map  = GridMaker.norm_views with LEAK_VIEWS=True, LEAK_NOISE=0
                 i.e. norm(to_unit(norm(view))), the Eq.9 + infer_fragments remap
  * model      = reconstruction_out/best_leak_down32_lpips.pt, the checkpoint
                 infer_fragments.py actually runs

Writes one CSV row per m as it finishes (checkpointed; safe to interrupt).

    python measure_mstar.py                 # clean, sigma_rec = 0
    SIGN_FLIP=1 python measure_mstar.py     # + the measured 47% fragment sign flip
    LEAK_NOISE=0.11 python measure_mstar.py # + a given fragment-space noise level
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reconstruction_test as R
from dataloader import CIFAR10BYOLClientData

CKPT      = "reconstruction_out/best_leak_down32_lpips.pt"
N_ORIG    = int(os.environ.get("N_ORIG", "512"))
M_LIST    = [1, 2, 4, 8, 16, 32, 64]
SIGN_FLIP = os.environ.get("SIGN_FLIP", "0") == "1"
OUT_CSV   = os.environ.get("OUT_CSV", "mstar_curve.csv")
BATCH     = 64

# sigma_rec = 0 unless asked otherwise: this is the demand-side curve, and the
# noise-side cost is accounted separately (it sets the fragment SUPPLY via T_eff).
R.LEAK_NOISE = float(os.environ.get("LEAK_NOISE", "0.0"))
R.LEAK_CONTAM_P = 0.0
assert R.LEAK_VIEWS, "LEAK_VIEWS must be on: the model was trained in fragment space"


def ssim_per_image(a, b):
    """reconstruction_test.ssim, reduced per image instead of over the batch."""
    import torch.nn.functional as F
    a, b = a.float(), b.float()
    c = a.shape[1]
    w1 = R._gaussian_window(11, 1.5, a.device).to(a.dtype)
    w = (w1[:, None] @ w1[None, :]).expand(c, 1, 11, 11).contiguous()
    mu1, mu2 = F.conv2d(a, w, padding=5, groups=c), F.conv2d(b, w, padding=5, groups=c)
    m1s, m2s, m12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = F.conv2d(a * a, w, padding=5, groups=c) - m1s
    s2 = F.conv2d(b * b, w, padding=5, groups=c) - m2s
    s12 = F.conv2d(a * b, w, padding=5, groups=c) - m12
    c1, c2 = 1e-4, 9e-4
    return (((2 * m12 + c1) * (2 * s12 + c2)) /
            ((m1s + m2s + c1) * (s1 + s2 + c2))).mean(dim=(1, 2, 3))


def build_inputs(device):
    """[N_ORIG,64,3,32,32] uint8 views + [N_ORIG,3,32,32] uint8 originals, client 0."""
    d = CIFAR10BYOLClientData(num_clients=5, classes_per_client=2, cid=0, batch_size=256,
                              data_dir="./data", num_workers=2, download=False,
                              data_fraction=1.0, strong_aug=R.STRONG_AUG)
    base = d.client_train_byol.base_dataset
    n = min(N_ORIG, len(base))
    views = np.empty((n, max(M_LIST), 3, 32, 32), dtype=np.uint8)
    origs = np.empty((n, 32, 32, 3), dtype=np.uint8)
    t0 = time.time()
    for i in range(n):
        img, _ = base[i]
        origs[i] = np.asarray(img, dtype=np.uint8)
        views[i] = R.pil_views(origs[i], max(M_LIST), seed=R.SEED * 1_000_003 + i)
        if i % 128 == 0:
            print(f"  views {i}/{n}  ({time.time()-t0:.0f}s)", flush=True)
    o = torch.from_numpy(origs).permute(0, 3, 1, 2).contiguous()
    return torch.from_numpy(views).to(device), o.to(device)


@torch.no_grad()
def main():
    device = R.get_device()
    model = R.build_model(R.MODEL).to(device).eval()
    model.load_state_dict(torch.load(CKPT, map_location=device))
    gm = R.GridMaker().to(device)
    lp = R._lpips_model(device)
    print(f"model={R.MODEL} ckpt={CKPT}  LEAK_NOISE={R.LEAK_NOISE}  SIGN_FLIP={SIGN_FLIP}")

    views_u8, orig_u8 = build_inputs(device)
    n = views_u8.shape[0]
    print(f"originals={n} (client-0 CIFAR-10 shard)  views/original={views_u8.shape[1]}")

    new = not os.path.exists(OUT_CSV)
    f = open(OUT_CSV, "a", newline="")
    wr = csv.writer(f)
    if new:
        wr.writerow(["m", "leak_noise", "sign_flip", "n", "ssim_mean", "ssim_median",
                     "lpips_mean", "lpips_median", "frac_ssim_gt_0.7", "frac_ssim_gt_0.6",
                     "frac_ssim_gt_0.5", "frac_lpips_lt_0.23", "frac_lpips_lt_0.3",
                     "frac_lpips_lt_0.4", "frac_lpips_lt_0.5"])
        f.flush()

    for m in M_LIST:
        ss, ll = [], []
        for i in range(0, n, BATCH):
            v = views_u8[i:i + BATCH, :m]
            if SIGN_FLIP:
                # measured on real fragments: ~47% arrive as photographic negatives,
                # applied in the pre-to_unit fragment space where the flip happens.
                vv = gm.norm(v.float().reshape(-1, 3, 32, 32)).reshape(v.shape[0], m, 3, 32, 32)
                s = torch.where(torch.rand(v.shape[0], m, 1, 1, 1, device=device) < 0.475,
                                -1.0, 1.0)
                x = gm.leak_views(vv * s)
            else:
                x = gm.norm_views(v)
            gt = gm.norm(orig_u8[i:i + BATCH])
            with torch.autocast("cuda", enabled=R.AMP and device.type == "cuda"):
                pred = model(x)
            p01, g01 = R.denormalize(pred.float()), R.denormalize(gt)
            ss.append(ssim_per_image(p01, g01).cpu())
            ll.append(lp(p01 * 2 - 1, g01 * 2 - 1).flatten().float().cpu())
        ss, ll = torch.cat(ss), torch.cat(ll)
        row = [m, R.LEAK_NOISE, int(SIGN_FLIP), n,
               f"{ss.mean():.4f}", f"{ss.median():.4f}",
               f"{ll.mean():.4f}", f"{ll.median():.4f}"] + \
              [f"{float((ss > t).float().mean()):.4f}" for t in (0.7, 0.6, 0.5)] + \
              [f"{float((ll < t).float().mean()):.4f}" for t in (0.23, 0.3, 0.4, 0.5)]
        wr.writerow(row); f.flush()
        print(f"m={m:3d}  SSIM {ss.mean():.3f} (>0.7: {float((ss>0.7).float().mean())*100:5.1f}%)   "
              f"LPIPS {ll.mean():.3f} (<0.3: {float((ll<0.3).float().mean())*100:5.1f}%)", flush=True)
    f.close()
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()

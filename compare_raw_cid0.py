"""
compare_raw_cid0.py -- score raw Eq. 9 fragments against client 0's real shard.

Same matcher as compare_cid0.py (blurred NCC over the participating pool, then
VGG LPIPS on the top-8 client-0 candidates), streamed over round_*.pt files
instead of inverter PNGs. Each tile is one sign-canonicalised, min-maxed
augmented view, so the LPIPS bins are a separate measurement from the
inverter's reconstruction score.

    ONLY_ROUND=50 python compare_raw_cid0.py
    FRAG_DIR=secagg_byol_fragments python compare_raw_cid0.py
"""
import glob
import os
import time

import lpips
import numpy as np
import torch
import torch.nn.functional as Fn
from torchvision.utils import make_grid, save_image

import compare_cid0 as C
import regroup_fragments as RG
import reconstruction_test as R

FRAG_DIR   = os.environ.get("FRAG_DIR", "fragments")
QA_OUT     = os.environ.get("QA_OUT", "cid0_compare_raw")
MAX_ROUNDS = int(os.environ.get("MAX_ROUNDS", "0"))   # 0 = every round file
ONLY_ROUND = int(os.environ.get("ONLY_ROUND", "0"))   # server round, 0 = all


def round_files():
    files = sorted(glob.glob(os.path.join(FRAG_DIR, "round_*.pt")))
    if ONLY_ROUND:
        name = f"round_{ONLY_ROUND:03d}.pt"
        files = [f for f in files if os.path.basename(f) == name]
        if not files:
            raise SystemExit(f"missing {name} under {FRAG_DIR}")
    elif MAX_ROUNDS > 0:
        files = files[:MAX_ROUNDS]
    if not files:
        raise SystemExit(f"no round_*.pt under {FRAG_DIR}")
    return files


def keep_best(imgs, lp, lp_j, best_lp, best_tiles=None):
    """Per client-0 image, keep the fragment with the lowest LPIPS so far."""
    order = torch.argsort(lp)
    _, first = np.unique(lp_j[order].numpy(), return_index=True)
    wins = order[torch.from_numpy(np.ascontiguousarray(first))]
    js = lp_j[wins]
    better = lp[wins] < best_lp[js]
    if not bool(better.any()):
        return
    js, wins = js[better], wins[better]
    if best_tiles is not None:
        best_tiles[js] = imgs[wins]
    best_lp[js] = lp[wins]


def montage(path, rows, tiles, gt):
    if len(rows) == 0:
        return
    pairs = []
    for j in rows:
        up = lambda im: Fn.interpolate(im[None], scale_factor=4, mode="nearest")[0]
        pairs.append(up(tiles[j]))
        pairs.append(up(gt[j]))
    grid = make_grid(torch.stack(pairs), nrow=8, padding=2)
    save_image(grid, path)


def bin_table(title, values, n_ref):
    print(f"\n{title}")
    print("  LPIPS bin     count")
    for lo, hi in zip(C.LP_BINS[:-1], C.LP_BINS[1:]):
        nb = int(((values >= lo) & (values < hi)).sum())
        print(f"  {lo:.2f}-{hi:.2f}    {nb:>8d}")
    print(f"  in bins {int(values.numel())} / {n_ref}")


def main():
    device = R.get_device()
    t0 = time.time()
    print(f"raw fragments: {FRAG_DIR}  -> montages {QA_OUT}/")
    print("each tile is one sign-canonicalised augmented view; "
          "LPIPS is against the unmodified client-0 image")
    print(f"dataset: {C.DATASET}  (num_clients={C.NUM_CLIENTS}, classes/client={C.CLS_PER}, "
          f"data_fraction={C.DATA_FRACTION})")
    data, targets = C.load_dataset()
    idx0 = C.client_indices(targets, C.TARGET_CID)
    idx1 = C.client_indices(targets, C.NULL_CID)
    cls0 = sorted(set(targets[idx0].tolist()))
    print(f"pool: {len(data)} imgs | client {C.TARGET_CID}: {len(idx0)} imgs across "
          f"{len(cls0)} classes | null client {C.NULL_CID}: "
          f"{len(set(targets[idx1].tolist()))} classes")

    full_f = C.ncc_feats(data, device)
    idx0_t = torch.from_numpy(np.ascontiguousarray(idx0)).to(device)
    gt0_f = full_f[idx0_t]
    gt1_f = full_f[torch.from_numpy(np.ascontiguousarray(idx1)).to(device)]
    gt0_imgs = data[torch.from_numpy(np.ascontiguousarray(idx0))]

    lp_model = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
    for p in lp_model.parameters():
        p.requires_grad_(False)

    n_img = len(idx0)
    best_lp = torch.full((n_img,), float("inf"))
    best_tiles = torch.zeros(n_img, *data.shape[1:])
    scores0, scores1, lp_all = [], [], []
    n_stored = n_usable = n_in0 = 0

    files = round_files()
    print(f"{len(files)} round files", flush=True)
    for fi, path in enumerate(files):
        tr = time.time()
        frags = RG._load_round(path)
        if frags.numel() == 0:
            print(f"  {os.path.basename(path)} empty", flush=True)
            continue
        n_stored += int(frags.shape[0])
        live = frags.flatten(1).std(1) >= RG.BLANK_STD
        frags = frags[live]
        if frags.numel() == 0:
            print(f"  {os.path.basename(path)} no tile above std {RG.BLANK_STD}", flush=True)
            continue
        imgs = RG.to_unit(frags)
        del frags
        n_usable += int(imgs.shape[0])

        rf = C.ncc_feats(imgs, device)
        _sc_full, idx_full = C.best_match(rf, full_f)
        n_in0 += int(torch.isin(idx_full, idx0_t).sum())
        sc0, _ = C.best_match(rf, gt0_f)
        sc1, _ = C.best_match(rf, gt1_f)
        scores0.append(sc0.detach().cpu())
        scores1.append(sc1.detach().cpu())
        cand = C.topk_ncc(rf, gt0_f, C.LPIPS_K)
        del rf, idx_full, sc0, sc1
        lp, lp_j = C.lpips_match(imgs, gt0_imgs, cand, lp_model, device)
        keep_best(imgs, lp, lp_j, best_lp, best_tiles)
        lp_all.append(lp)
        under = int((lp < 0.30).sum())
        print(f"  {os.path.basename(path)}  usable {int(live.sum())}/{int(live.numel())}  "
              f"lp<0.30 {under}  {time.time()-tr:.0f}s", flush=True)
        del imgs, cand, lp, lp_j

    if n_usable == 0:
        raise SystemExit("no usable fragments")

    sc0 = torch.cat(scores0)
    sc1 = torch.cat(scores1)
    lp = torch.cat(lp_all)
    matched = best_lp < float("inf")
    img_lp = best_lp[matched]
    chance = len(idx0) / len(data) * 100

    print(f"\nusable fragments: {n_usable}/{n_stored}  "
          f"(std >= {RG.BLANK_STD})")
    print(f"full-set nearest match in client-0 SET : {n_in0/n_usable*100:5.1f}%  "
          f"(chance {chance:.0f}%)")
    print(f"\nbest NCC vs client 0  : median {float(sc0.median()):.3f}  "
          f"p90 {float(sc0.quantile(.9)):.3f}")
    print(f"best NCC vs client 1  : median {float(sc1.median()):.3f}  "
          f"p90 {float(sc1.quantile(.9)):.3f}  (null)")
    print(f"fragments matching client 0 better than client 1: "
          f"{float((sc0 > sc1).float().mean())*100:.1f}%")

    print(f"\nLPIPS(fragment, best client-0 match): median {float(lp.median()):.3f}  "
          f"p10 {float(lp.quantile(.1)):.3f}  p90 {float(lp.quantile(.9)):.3f}  "
          f"(lower = better)")
    print("  fragment-level @LPIPS<t:")
    for t in (0.10, 0.15, 0.20, 0.25, 0.30):
        d = int((lp < t).sum())
        print(f"     <{t:.2f}: {d:>8d}/{n_usable} ({d/n_usable*100:5.2f}%)")
    print(f"  image-level (best fragment per client-0 image), "
          f"{int(matched.sum())}/{n_img} images matched:")
    for t in (0.10, 0.15, 0.20, 0.25, 0.30):
        d = int((img_lp < t).sum())
        print(f"     <{t:.2f}: {d:>5d}/{n_img} ({d/n_img*100:5.2f}%)")

    bin_table("fragment-level LPIPS bins", lp, n_usable)
    bin_table("image-level LPIPS bins (best fragment per matched image)", img_lp, n_img)

    js = matched.nonzero(as_tuple=True)[0]
    ss = C.ssim_pairs(best_tiles[js], gt0_imgs[js], device)
    print(f"\nSSIM(best fragment, client-0 image): median {float(ss.median()):.3f}  "
          f"p10 {float(ss.quantile(.1)):.3f}  p90 {float(ss.quantile(.9)):.3f}")

    os.makedirs(QA_OUT, exist_ok=True)
    order = torch.argsort(best_lp)
    order = order[matched[order]]
    mid = len(order) // 2
    rows = {
        "lpips_best": order[:32].tolist(),
        "median": order[max(0, mid - 16):mid + 16].tolist(),
        "lpips_worst": order[-32:].tolist(),
    }
    near = (best_lp - C.LP_GATE).abs()
    near[~matched] = float("inf")
    boundary = torch.argsort(near)[:32]
    boundary = boundary[torch.argsort(best_lp[boundary])]
    rows[f"lpips_at_{C.LP_GATE:.2f}"] = boundary.tolist()
    for tag, sel in rows.items():
        montage(os.path.join(QA_OUT, f"match_{tag}.png"), sel, best_tiles, gt0_imgs)
    for lo, hi in zip(C.LP_BINS[:-1], C.LP_BINS[1:]):
        m = ((best_lp >= lo) & (best_lp < hi)).nonzero(as_tuple=True)[0]
        if m.numel() == 0:
            continue
        pick = torch.linspace(0, m.numel() - 1, min(32, m.numel())).long()
        montage(os.path.join(QA_OUT, f"match_lpbin_{lo:.2f}-{hi:.2f}.png"),
                m[pick].tolist(), best_tiles, gt0_imgs)
    print(f"\nmontages (raw tile | client-0 image) -> {QA_OUT}/")
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

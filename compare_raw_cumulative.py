"""
compare_raw_cumulative.py -- client-0 matches as rounds accumulate.

Same matcher as compare_raw_cid0.py (sign canon, blank filter, blurred NCC,
VGG LPIPS on the top-8 client-0 candidates). Rounds are taken in server order.
After each file the curve records how many client-0 images have a best tile
so far under each LPIPS gate. One pass: each tile is scored once, and the
prefix totals are that running minimum.

    python compare_raw_cumulative.py

Writes cid0_raw_curve/cumulative.csv and cumulative.png. A state file next
to them lets a killed run continue from the last finished round.
"""
import csv
import glob
import os
import time

import lpips
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import compare_cid0 as C
import compare_raw_cid0 as RAW
import regroup_fragments as RG
import reconstruction_test as R

FRAG_DIR = os.environ.get("FRAG_DIR", "fragments")
OUT_DIR = os.environ.get("OUT_DIR", "cid0_raw_curve")
THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30)
PLOT_GATES = (0.10, 0.20, 0.30)


def _key(t):
    return f"{t:.2f}"


COLUMNS = (
    ["through_round", "n_files", "n_stored", "n_usable", "img_matched"]
    + [f"frag_lt_{_key(t)}" for t in THRESHOLDS]
    + [f"img_lt_{_key(t)}" for t in THRESHOLDS]
)


def round_files():
    files = sorted(glob.glob(os.path.join(FRAG_DIR, "round_*.pt")))
    if not files:
        raise SystemExit(f"no round_*.pt under {FRAG_DIR}")
    return files


def server_round(path):
    return int(os.path.splitext(os.path.basename(path))[0].split("_")[1])


def snapshot(through, n_files, n_stored, n_usable, best_lp, frag_lt):
    matched = best_lp < float("inf")
    row = {
        "through_round": through,
        "n_files": n_files,
        "n_stored": n_stored,
        "n_usable": n_usable,
        "img_matched": int(matched.sum()),
    }
    for t in THRESHOLDS:
        row[f"frag_lt_{_key(t)}"] = frag_lt[t]
        row[f"img_lt_{_key(t)}"] = int((best_lp < t).sum())
    return row


def write_csv(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def write_plot(path, rows, n_img):
    xs = [int(r["through_round"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for t in PLOT_GATES:
        ys = [int(r[f"img_lt_{_key(t)}"]) / n_img * 100 for r in rows]
        ax.plot(xs, ys, label=f"LPIPS < {t:.2f}")
    ax.set_xlabel("through server round")
    ax.set_ylabel("% of client 0")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.set_title("Images with a raw fragment this close, rounds 2..k")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def load_resume(csv_path, state_path):
    if not (os.path.exists(csv_path) and os.path.exists(state_path)):
        return [], None
    with open(csv_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if not rows or int(rows[-1]["through_round"]) != int(state["through_round"]):
        print("resume state does not match the csv; starting over", flush=True)
        return [], None
    return rows, state


def main():
    device = R.get_device()
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "cumulative.csv")
    png_path = os.path.join(OUT_DIR, "cumulative.png")
    state_path = os.path.join(OUT_DIR, "state.pt")

    print(f"cumulative raw matches: {FRAG_DIR} -> {csv_path}")
    data, targets = C.load_dataset()
    idx0 = C.client_indices(targets, C.TARGET_CID)
    n_img = len(idx0)
    print(
        f"dataset: {C.DATASET} classes/client={C.CLS_PER} "
        f"client {C.TARGET_CID}: {n_img} images",
        flush=True,
    )
    if (
        C.DATASET == "cifar100"
        and C.CLS_PER == 20
        and C.NUM_CLIENTS == 5
        and C.DATA_FRACTION == 1
        and n_img != 10000
    ):
        raise SystemExit(f"cifar100 client 0 has {n_img} images, expected 10000")

    full_f = C.ncc_feats(data, device)
    gt0_f = full_f[torch.from_numpy(np.ascontiguousarray(idx0)).to(device)]
    gt0_imgs = data[torch.from_numpy(np.ascontiguousarray(idx0))]
    del full_f

    lp_model = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
    for p in lp_model.parameters():
        p.requires_grad_(False)

    rows, state = load_resume(csv_path, state_path)
    if state is None:
        best_lp = torch.full((n_img,), float("inf"))
        frag_lt = {t: 0 for t in THRESHOLDS}
        n_stored = n_usable = 0
        done = 0
    else:
        best_lp = state["best_lp"]
        frag_lt = state["frag_lt"]
        n_stored = int(state["n_stored"])
        n_usable = int(state["n_usable"])
        done = int(state["through_round"])
        print(f"resuming after round {done} ({len(rows)} rows)", flush=True)

    files = [f for f in round_files() if server_round(f) > done]
    print(f"{len(files)} round files left", flush=True)
    for path in files:
        tr = time.time()
        rnd = server_round(path)
        frags = RG._load_round(path)
        n_here = int(frags.shape[0]) if frags.numel() else 0
        n_stored += n_here
        if n_here:
            live = frags.flatten(1).std(1) >= RG.BLANK_STD
            frags = frags[live]
        if frags.numel():
            imgs = RG.to_unit(frags)
            del frags
            n_usable += int(imgs.shape[0])
            rf = C.ncc_feats(imgs, device)
            cand = C.topk_ncc(rf, gt0_f, C.LPIPS_K)
            del rf
            lp, lp_j = C.lpips_match(imgs, gt0_imgs, cand, lp_model, device)
            RAW.keep_best(imgs, lp, lp_j, best_lp)
            for t in THRESHOLDS:
                frag_lt[t] += int((lp < t).sum())
            del imgs, cand, lp, lp_j

        row = snapshot(rnd, len(rows) + 1, n_stored, n_usable, best_lp, frag_lt)
        if rows:
            for t in THRESHOLDS:
                key = f"img_lt_{_key(t)}"
                if int(row[key]) < int(rows[-1][key]):
                    raise RuntimeError(f"{key} decreased at round {rnd}")
        rows.append(row)
        write_csv(csv_path, rows)
        write_plot(png_path, rows, n_img)
        torch.save({
            "through_round": rnd,
            "best_lp": best_lp,
            "frag_lt": frag_lt,
            "n_stored": n_stored,
            "n_usable": n_usable,
        }, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)
        print(f"  through {rnd:3d}  "
              f"img<0.20 {row['img_lt_0.20']:>5d}  "
              f"img<0.30 {row['img_lt_0.30']:>5d}  "
              f"frag<0.30 {row['frag_lt_0.30']:>6d}  "
              f"{time.time()-tr:.0f}s", flush=True)

    print(f"\n{len(rows)} prefixes -> {csv_path}")
    print(f"curve -> {png_path}")
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

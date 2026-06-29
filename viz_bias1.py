"""
viz_bias1.py -- visualize raw LOKI fragments with bias_count == 1, so we can
eyeball how corrupted these "clean single-image" reconstructions actually are.

No model involved: just load the stored fragments, map each to [0,1] with the
project's canonical per-image min-max (infer_fragments.to_unit / Loki._to_unit)
and tile them. Each row is one bin; each column a different bias_count==1 round.
"""
import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

FRAG_DIR = "fragments"
OUT = os.path.join("reconstruction_out", "bias1_only_fragments.png")
N_BINS = 8          # rows
N_FRAGS = 8         # columns (fragments per bin)


def all_bias1_rounds(bin_dir):
    """Rounds for a bin, but only if EVERY round has bias_count == 1.

    Returns the sorted round list when the whole bin is bias_count==1,
    else [] (any 0/contaminated round disqualifies the bin).
    """
    ic = os.path.join(bin_dir, "info.csv")
    if not os.path.exists(ic):
        return []
    rounds, counts = [], []
    for row in csv.DictReader(open(ic)):
        rounds.append(int(row["round"]))
        counts.append((row.get("bias_count") or "").strip())
    if not counts or any(c != "1" for c in counts):
        return []
    return sorted(rounds)


def to_unit(img):
    lo = img.amin(dim=(-3, -2, -1), keepdim=True)
    hi = img.amax(dim=(-3, -2, -1), keepdim=True)
    return (img - lo) / (hi - lo + 1e-8)


def main():
    bins = []
    for b in sorted(glob.glob(os.path.join(FRAG_DIR, "bin_*"))):
        rounds = all_bias1_rounds(b)
        if len(rounds) >= N_FRAGS:
            bins.append((b, rounds[:N_FRAGS]))
        if len(bins) >= N_BINS:
            break

    if not bins:
        print("no all-bias_count==1 bins with enough rounds")
        return

    fig, axes = plt.subplots(len(bins), N_FRAGS,
                             figsize=(N_FRAGS * 1.3, len(bins) * 1.4))
    axes = np.atleast_2d(axes)
    for r, (b, rounds) in enumerate(bins):
        for c, rnd in enumerate(rounds):
            ax = axes[r][c]
            ax.axis("off")
            p = os.path.join(b, f"round_{rnd:03d}.pt")
            frag = torch.load(p, map_location="cpu", weights_only=False)
            img = to_unit(frag).permute(1, 2, 0).numpy()
            ax.imshow(img)
            if r == 0:
                ax.set_title(f"frag {c+1}", fontsize=8)
            if c == 0:
                ax.set_title("")  # keep row label clean
        axes[r][0].set_ylabel(os.path.basename(b), fontsize=7, rotation=0,
                              labelpad=24, va="center")
        axes[r][0].axis("on")
        axes[r][0].set_xticks([]); axes[r][0].set_yticks([])

    fig.suptitle("Raw LOKI fragments (bins where EVERY round bias_count==1)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT} | {len(bins)} bins x {N_FRAGS} fragments")


if __name__ == "__main__":
    main()

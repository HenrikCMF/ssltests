"""
infer_fragments.py -- run the trained reconstructor on REAL LOKI fragments.

For every FULL bin (a folder holding ~100 across-round fragments of one original),
load all of its fragments, feed them to the trained reconstructor (best.pt) and
save the reconstructed original. One reconstruction PNG per bin.

Pipeline per bin:
  1. A "full" bin is one with FULL_COUNT round_*.pt fragments (the folder then
     has FULL_COUNT+1 entries counting info.csv -- the "100 items" of a full bin).
  2. Load (up to MAX_VIEWS of) its round_*.pt fragments. Each is [3,32,32] float.
  3. Map fragment -> [0,1] via per-image min-max (attacks.Loki._to_unit, the
     project's canonical fragment->image mapping), then CIFAR-normalize -> the
     exact input space the model trained on.
  4. best.pt is the SetTransformer: it consumes the view *set* [1,N,3,32,32]
     directly (each view is a token; it is permutation/count-invariant), NOT a
     pre-tiled grid. We feed the set and save the [3,32,32] reconstruction.

Optionally (SAVE_MONTAGE) also dump the 10x10 concatenation of the input
fragments next to each reconstruction, for visual inspection.
"""
import glob
import os

import torch
from torchvision.utils import make_grid, save_image

import reconstruction_test as R
#FRAG_DIR=fragments_clustered MIN_VIEWS=4 RECON_SUBDIR=cluster_recons python infer_fragments.py

# FRAG_DIR / OUT subdir / min-views are env-overridable so the same script can run
# on the raw leak bins (defaults) or on re-identified clusters from cluster_fragments
# (e.g. FRAG_DIR=fragments_clustered MIN_VIEWS=4 RECON_SUBDIR=cluster_recons python ...).
FRAG_DIR    = os.environ.get("FRAG_DIR", "fragments")
OUT_DIR     = os.path.join(R.OUT_DIR, os.environ.get("RECON_SUBDIR", "fragment_recons"))
CKPT        = os.environ.get("CKPT", "reconstruction_out/best_leak_down32_0.05noise.pt")#"reconstruction_out/best_leak_down_tinyimage_og.pt")   # inverter; env-overridable
FULL_COUNT  = int(os.environ.get("MIN_VIEWS", "99"))  # min round_*.pt to reconstruct a bin
MAX_VIEWS   = R.NUM_PERTURB    # cap views fed (the model trains on up to this many)
BATCH_BINS  = 64               # bins reconstructed per forward pass
DROP_BLANKS = True             # drop empty/saturated rounds (a real bin is ~8% blank tiles)
BLANK_STD   = 0.05             # raw-space per-fragment std below this == blank (real ~0.5)
SAVE_MONTAGE = False            # also save the 10x10 input-fragment grid per bin
GRID        = R.GRID           # 10 -> 10x10 montage

_MEAN = torch.tensor(R.CIFAR_MEAN).view(1, 3, 1, 1)
_STD  = torch.tensor(R.CIFAR_STD).view(1, 3, 1, 1)


def to_unit(img):
    """Per-image min-max to [0,1] -- matches attacks.Loki._to_unit."""
    lo = img.amin(dim=(-3, -2, -1), keepdim=True)
    hi = img.amax(dim=(-3, -2, -1), keepdim=True)
    return (img - lo) / (hi - lo + 1e-8)


def full_bins(frag_dir):
    """Bins with exactly FULL_COUNT round_*.pt fragments."""
    out = []
    for b in sorted(glob.glob(os.path.join(frag_dir, "bin_*"))):
        if len(glob.glob(os.path.join(b, "round_*.pt"))) >= FULL_COUNT:
            out.append(b)
    return out


def load_views(bin_dir):
    """A bin's fragments as a model-space set [N,3,32,32] (N<=MAX_VIEWS).

    Empty/saturated rounds (blank tiles, ~8% of a real bin and up to 70% in some)
    are dropped: detected on the RAW fragment (per-fragment std ~0), which the
    clean-augmented training set never contained. The model is count-invariant,
    so feeding only the informative views is strictly better than diluting the
    attention pool with blanks. Never drops the last surviving view."""
    paths = sorted(glob.glob(os.path.join(bin_dir, "round_*.pt")))[:MAX_VIEWS]
    frags = torch.stack([torch.load(p, map_location="cpu", weights_only=False) for p in paths])
    if DROP_BLANKS:
        keep = frags.flatten(1).std(dim=1) >= BLANK_STD     # raw std, pre-normalization
        if bool(keep.any()):
            frags = frags[keep]
    v01 = to_unit(frags)                       # [N,3,32,32] in [0,1]
    return (v01 - _MEAN) / _STD                # CIFAR-normalized -> model input


def main():
    device = R.get_device()
    model = R.build_model(R.MODEL).to(device).eval()
    model.load_state_dict(torch.load(CKPT, map_location=device))
    os.makedirs(OUT_DIR, exist_ok=True)

    bins = full_bins(FRAG_DIR)
    print(f"{len(bins)} full bins (>= {FULL_COUNT} fragments) -> {OUT_DIR}/")

    done = 0
    for i in range(0, len(bins), BATCH_BINS):
        chunk = bins[i:i + BATCH_BINS]
        sets = [load_views(b) for b in chunk]
        # Blank-dropping leaves bins with uneven view counts; pad each up to the
        # chunk max by repeating its OWN views (the set model is count-invariant,
        # so repeats just re-weight) rather than truncating everyone to the min.
        maxn = max(v.shape[0] for v in sets)
        def _pad(v):
            if v.shape[0] >= maxn:
                return v[:maxn]
            reps = (maxn + v.shape[0] - 1) // v.shape[0]
            return v.repeat(reps, 1, 1, 1)[:maxn]
        batch = torch.stack([_pad(v) for v in sets]).to(device)   # [B,maxn,3,32,32]
        with torch.no_grad(), torch.autocast(device.type, enabled=R.AMP and device.type == "cuda"):
            preds = model(batch)                       # [B,3,32,32]
        recons = R.denormalize(preds.float()).cpu()
        for b, views, recon in zip(chunk, sets, recons):
            name = os.path.basename(b)
            save_image(recon, os.path.join(OUT_DIR, f"{name}.png"))
            if SAVE_MONTAGE:
                grid = make_grid(R.denormalize(views), nrow=GRID, padding=1)
                save_image(grid, os.path.join(OUT_DIR, f"{name}_input.png"))
            done += 1
        print(f"  {done}/{len(bins)}", flush=True)

    print(f"done. {done} reconstructions in {OUT_DIR}/")


if __name__ == "__main__":
    main()

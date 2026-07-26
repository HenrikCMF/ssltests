"""
reconstruction_test.py -- offline image-reconstruction engine for the LOKI attack.

Idea (see the discussion around attacks.Loki): a LOKI bin folder accumulates ~100
augmented *views* of (probably) one original image across rounds. If we know the
augmentation pipeline, we can train a model to invert it: feed it the stack of
views and have it regress the single clean original.

This is the supervised proof-of-concept for that inverter, trained on data we
control:

  1. Load CIFAR-100 (downloaded on first run).
  2. Split 99.9/0.1 into train/test.
  3. PRECOMPUTE, once and in parallel, `NUM_PERTURB` *bit-exact torchvision/PIL*
     perturbations per original (the same transform the federated clients use in
     dataloader._build_byol_transform, minus the final ToTensor/Normalize which is
     applied identically on the GPU at train time). Cached to a uint8 memmap so
     training just reads it -- no per-step augmentation, which was the bottleneck.
  4. Tile the views into a square grid and push it through a CNN ending in a linear
     layer with one neuron per ground-truth pixel-channel (3*32*32), reshaped back.
  5. Train with an SSIM loss (1 - SSIM). Mixed precision + channels_last.

The precomputed pixels are identical to what the real clients produce, so the
inverter is trained on exactly the distribution it will face on real fragments.
The model is swappable (build_model) for a transformer later.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

# ---------------------------------------------------------------------------- #
# Config
# ---------------------------------------------------------------------------- #
DATA_DIR      = "./data"
CACHE_DIR     = "./recon_cache"
# --- Resolution / source-dataset mode --------------------------------------- #
# The inverter's TRAINING data is chosen by these toggles. Enable AT MOST one of
# UPSCALE / DOWNSCALE / SVHN; all off = CIFAR-100 at its native 32x32.
#   (none)     CIFAR-100, native 32x32.
#   UPSCALE    CIFAR-100 32x32 LANCZOS-upscaled to UPSCALE_SIZE *before* perturbing.
#              Targets are band-limited (no genuine detail above 32px), so the inverter
#              learns to emit smooth ~32px-of-detail images -> blurry 64px recons.
#   DOWNSCALE  Tiny ImageNet native 64x64 LANCZOS-downscaled to DOWNSCALE_SIZE *before*
#              perturbing. Targets carry real (decimated, not invented) detail, so the
#              inverter trains to output crisp small images of true 64px content.
#   SVHN       SVHN (Street View House Numbers) train split, native 32x32, NO resize. A
#              source-INDEPENDENT 32px control: SVHN is Google Street View, not the 80M
#              Tiny Images family CIFAR comes from, so training here and testing on
#              CIFAR-10/100 probes cross-collection transfer without any resize artifact.
# Whichever is on, IMG_SIZE drives every downstream array/model automatically.
UPSCALE        = False
UPSCALE_SIZE   = 64
DOWNSCALE      = True
DOWNSCALE_SIZE = 32
SVHN           = False        # train the inverter on SVHN (native 32x32, source-independent control)
assert sum([UPSCALE, DOWNSCALE, SVHN]) <= 1, "enable at most one of UPSCALE / DOWNSCALE / SVHN"
_RESIZE  = UPSCALE or DOWNSCALE                    # source native size != IMG_SIZE -> LANCZOS resize (SVHN is native 32)
IMG_SIZE = (UPSCALE_SIZE if UPSCALE else
            DOWNSCALE_SIZE if DOWNSCALE else 32)   # effective working resolution
NUM_PERTURB   = 100          # views per original (one LOKI bin's across-round stack)
GRID          = 10           # NUM_PERTURB must equal GRID*GRID
TRAIN_FRAC    = 0.999        # 99.9/0.1 split -> ~50 test originals (cheap per-epoch eval)
TRAIN_SUBSET  = None         # set to an int to cap training originals (faster smoke run)
BATCH_SIZE    = 256          # originals per step; each expands to BATCH*NUM_PERTURB views
EPOCHS        = 500
LR            = 3e-4         # transformer-safe; 1e-3 held near-peak for 100+ epochs blew up
WARMUP_EPOCHS = 3            # linear LR warmup (transformer stability), then cosine decay
GRAD_CLIP     = 1.0         # max grad-norm; prevents the late-epoch NaN blow-up
WEIGHT_DECAY  = 0.05         # AdamW decay (excl. norms/biases); regularizes the transformer
DROPOUT       = 0.1          # dropout inside the transformer enc/dec layers
# Each step trains on a random VIEW-COUNT subset so the model is robust to the
# wide spread of token counts it meets at inference. Real LOKI clusters are NOT
# ~100 views: they run from a handful to ~99, median ~15 (see cluster_purity.py),
# and the old fixed 64 left the model out-of-distribution on the common small
# clusters. The per-step count is drawn LOG-UNIFORMLY in [VIEWS_MIN, VIEWS_MAX] so
# small counts (where most real clusters live) are densely covered while large
# counts still appear regularly. Set VIEWS_MIN == VIEWS_MAX for a fixed count.
VIEWS_MIN     = 4
VIEWS_MAX     = 99
PERCEPTUAL_W  = 0.1          # weight of the VGG feature-matching term (0 = off); fights blur
# --- LPIPS training objective ----------------------------------------------- #
# LPIPS_LOSS trains the inverter on the LPIPS perceptual DISTANCE (Zhang et al. 2018:
# calibrated linear weights on deep features) instead of the SSIM+VGG combo. A small L1
# anchor is kept because pure LPIPS lets global colour/brightness drift (LPIPS is
# near-invariant to it).
# BACKBONE: use "vgg" at these small resolutions. AlexNet's stride-4 11x11 conv1 leaves
# almost no feature map at 32px, so its LPIPS dynamic range collapses (independent images
# score ~0.06, not ~0.5) -> tiny, meaningless distances that look great while recons are
# bad. VGG's gentle 3x3/stride-1 convs keep ~5x the range at 32px (independent ~0.29) and
# are the better optimization target. NOTE: compare_cid0 measures with alex@64, so the
# training LPIPS numbers here are NOT directly comparable to its scores.
LPIPS_LOSS    = True        # train on LPIPS instead of SSIM+VGG
LPIPS_NET     = "vgg"        # LPIPS backbone: "vgg" (real range at 32px) or "alex" (collapses <64px)
LPIPS_W       = 1.0          # weight of the LPIPS term
LPIPS_L1_W    = 0.1          # L1 anchor weight on colour/low-freq (pure LPIPS drifts there)
AMP           = True         # mixed precision
# --- Throughput knobs (pure speed; do not change the training math) ---------- #
# COMPILE    torch.compile the model. The per-view stem + Transformer fuse well; the
#            only dynamic axis is the per-step view count N, so we mark_dynamic dim 1
#            (see call_model) to compile ONCE instead of recompiling for every k.
# PREFETCH   overlap the CPU memmap view-gather + host->device copy of the NEXT batch
#            with the current batch's GPU compute (a background thread). The old loader
#            was fully synchronous -- ~79 MB gathered per step with the GPU idle.
# DUAL_GPU   data-parallel across cuda:0 (primary, RTX 5090) + cuda:1 (weaker card).
#            The batch's originals are split UNEVENLY -- GPU1 gets GPU1_FRAC of them --
#            because cuda:1 is far slower; giving it a small shard keeps it from being
#            the straggler the step waits on, and offloading that shard also frees VRAM
#            on cuda:0. Grads from the two shards are combined size-weighted (so it is
#            the exact same gradient as one big batch) on cuda:0, the single optimizer
#            steps there, and the updated weights are copied back to the replica. Only
#            cuda:0's model is ever checkpointed/evaluated.
COMPILE       = True
PREFETCH      = True
# DUAL_GPU measured a NET LOSS on this box (RTX 5090 + RTX 4060 Ti): the 4060 Ti is ~6x
# slower and sits on PCIe 4.0 x8, so the per-step grad+weight sync (~308 MB ~19 ms) plus
# its oversized shard-time outweigh its compute contribution -- GPU0 dropped to ~50% util
# and the epoch got slower. Left as an off-by-default toggle; the win here is compile +
# prefetch on the 5090 alone. Only revisit dual with a closely-matched second GPU.
DUAL_GPU      = False
GPU1_FRAC     = 0.20         # fraction of each batch's originals sent to the weaker GPU1 (if DUAL_GPU)
PRECOMPUTE_WORKERS = 24
FORCE_PRECOMPUTE = True     # ignore cache and regenerate the views
SEED          = 12345
MODEL         = "settr"      # "settr" (view=token transformer), "set", or "cnn"
OUT_DIR       = "reconstruction_out"

# --- LOKI leak-in-the-loop (match the REAL fragment input space) ------------- #
# Today the model trains on clean augmented views: norm(view01). But infer_fragments
# feeds it norm(to_unit(fragment)), and a clean-bin fragment is max-abs(norm(view01)).
# Since to_unit is scale-invariant the max-abs cancels, so the real inference input is
#   norm( to_unit( norm(view01) ) )  -- a deterministic color-space remap the clean
# training never saw. LEAK_VIEWS applies that same operator to the training views so
# the model learns to invert what it will actually be fed. The two extra knobs model
# the NON-deterministic corruption of a real federated leak (multi-step real-loss
# training, imperfect binning); leave them at 0 to isolate the color-space fix.
#
# NOTE: domain-randomizing these per step with GAUSSIAN noise was tried and reverted
# -- it lifted oracle clusters (+4.7) but REGRESSED the real pipeline (24->21%); the
# real corruption is binning blends / few-view, not gaussian. See FINDINGS.md.
# UPDATE: that premise no longer holds once the federated run uses DP_MODE. DP injects
# a genuinely Gaussian perturbation into the transmitted update (client._apply_dp), so
# the real fragments now carry Gaussian corruption on TOP of the binning/few-view kind.
# LEAK_NOISE below is re-enabled to ~match that DP noise. It is NOT equal to DP_NOISE:
# DP_NOISE (sigma) is a RELATIVE fraction of a layer's update norm, whereas LEAK_NOISE is
# an ABSOLUTE std in the Eq.9 max-abs-1 fragment space. The link (see dp_noise_to_leak_noise
# below) goes through the FC1 WEIGHT update LOKI reads -- its RMS fc1w_rms and the Eq.9
# per-bin normalizer m_fired:
#   LEAK_NOISE ~= sqrt(K) * sigma * fc1w_rms / median(m_fired).
# CAUTION: do NOT use the client's logged "eff.abs.noise_std" (~430) here -- that is the
# dominant-NORM layer (FC1 bias/conv), a DIFFERENT layer LOKI never reconstructs from.
# fc1w_rms and m_fired are both logged by a real DP round (server LOKI reconstruct line).
# m_fired spans ~100x across bins, so one scalar LEAK_NOISE matches only the median bin;
# SWEEP around the computed value and judge by real-fragment recon.
LEAK_VIEWS    = True         # train/eval on LOKI-leak-style views (match real fragments)
LEAK_NOISE    = 0.14          # ~matches DP_NOISE=0.05; sweep {0.02,0.05,0.1} pre-to_unit
LEAK_CONTAM_P = 0.0           # per-view prob. of blending a decoy view (bin contamination)
LEAK_CONTAM_A = 0.5           # decoy blend weight when a view is contaminated


def dp_noise_to_leak_noise(sigma, fc1w_rms, m_fired, num_clients=1):
    """Convert a real DP run's FC1-weight noise into the equivalent sandbox LEAK_NOISE.

    Maps the weight-space DP perturbation (client._apply_dp) to the image-space
    LEAK_NOISE applied in leak_views, so both land the same noise-to-signal ratio
    on the pre-to_unit fragment that LOKI's Eq.9 produces.

    IMPORTANT: this must use the FC1 *weight* layer LOKI reconstructs from, NOT the
    client's logged "eff.abs.noise_std". That log is for the dominant-NORM layer
    (the FC1 bias / conv, d=fc_size) -- a *different* layer LOKI never reads, whose
    absolute noise (~430) is unrelated to the weight-gradient signal.

    Args:
      sigma    : DP_NOISE, the per-layer relative noise fraction (e.g. 0.1).
      fc1w_rms : RMS of the FC1 WEIGHT update = ||upd_w|| / sqrt(numel). The DP
                 noise on this layer had absolute per-coord std ~= sigma * fc1w_rms.
                 Printed by the server LOKI reconstruct line as "fc1w_rms".
      m_fired  : Eq.9 per-bin normalizer = max-abs of a fired bin's weight-update
                 row (use the MEDIAN over fired bins, "m_fired median" in the log).
                 reconstruct() divides each row by this, so the clean fragment has
                 max-abs 1 and the noise std becomes (sigma*fc1w_rms)/m_fired there.
      num_clients: K participating clients per round. Each adds independent noise
                 into every column LOKI reads while the de-aggregated signal is one
                 client's, so summed noise std ~ sqrt(K) * (sigma*fc1w_rms).

    Returns the LEAK_NOISE matching the MEDIAN fired bin. Because m_fired spans
    ~100x across bins, dim bins see far more relative noise than this and bright
    bins far less -- a single scalar LEAK_NOISE cannot capture that spread, which
    is itself a reason sandbox<->DP results diverge.
    """
    import math
    if m_fired <= 0:
        raise ValueError("m_fired must be > 0 (no fired bins?)")
    return math.sqrt(num_clients) * sigma * fc1w_rms / m_fired
# Client augmentation-strength toggle. MUST match the STRONG_AUG flag of the
# federated run whose fragments will be inverted: it ~doubles the perturbation
# strength of PRECOMPUTE_TF (see below) so the inverter trains on the exact view
# distribution the real clients produce. Strong/weak views and checkpoints are
# kept in separate files (via _AUG_TAG) so flipping this never clobbers the other.
STRONG_AUG    = False
_RES_TAG      = (f"_up{IMG_SIZE}" if UPSCALE else              # keep each mode's ckpt/cache
                 f"_down{IMG_SIZE}" if DOWNSCALE else            # files separate & non-clobbering
                 f"_svhn{IMG_SIZE}" if SVHN else "")
_LOSS_TAG     = "_lpips" if LPIPS_LOSS else ""                  # keep LPIPS-trained ckpts separate
_AUG_TAG      = "_strong" if STRONG_AUG else ""                # strong-aug views/ckpts kept separate
CKPT_NAME     = ("best_leak" if LEAK_VIEWS else "best") + _RES_TAG + _LOSS_TAG + _AUG_TAG + ".pt"  # don't clobber others

# CIFAR normalization (must match the federated pipeline exactly).
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

assert GRID * GRID == NUM_PERTURB, "NUM_PERTURB must be a perfect square == GRID*GRID"

# Exact client augmentation MINUS the final ToTensor/Normalize. We keep the output
# as a uint8 PIL image (same pixels the client feeds ToTensor) and do /255 +
# normalize on the GPU -- mathematically identical to ToTensor()+Normalize().
#
# STRONG_AUG (set above) mirrors dataloader.BYOLClientData's toggle exactly, along
# the same knobs it scales for the CIFAR/CIFAR-100 client path: crop min-scale
# 0.2->0.1, ColorJitter (0.8,0.8,0.8,0.2)->(1.6,1.6,1.6,0.4), grayscale p 0.2->0.4.
# Horizontal flip stays at p=0.5. (No GaussianBlur in either mode here, matching
# this pipeline's existing CIFAR-style transform.)
if STRONG_AUG:
    _COLOR_JITTER = T.ColorJitter(1.6, 1.6, 1.6, 0.4)
    _CROP_MIN_SCALE, _GRAYSCALE_P = 0.1, 0.4
else:
    _COLOR_JITTER = T.ColorJitter(0.8, 0.8, 0.8, 0.2)
    _CROP_MIN_SCALE, _GRAYSCALE_P = 0.2, 0.2
PRECOMPUTE_TF = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(_CROP_MIN_SCALE, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomApply([_COLOR_JITTER], p=0.8),
    T.RandomGrayscale(p=_GRAYSCALE_P),
])
loss_func = nn.MSELoss()

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------- #
# PIL view generation (used by both the parallel precompute and the test set)
# ---------------------------------------------------------------------------- #
def resize_orig(orig_hwc: np.ndarray) -> np.ndarray:
    """[H,W,3] uint8 original -> [IMG_SIZE,IMG_SIZE,3] uint8 via LANCZOS (best-quality PIL
    resample, up or down). Resizes the source before perturbing and builds the matching
    ground truth the inverter must output. UPSCALE goes 32->64, DOWNSCALE 64->32; a no-op
    when the source is already IMG_SIZE (plain CIFAR-100)."""
    from PIL import Image
    return np.asarray(
        Image.fromarray(orig_hwc).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS),
        dtype=np.uint8)


def pil_views(orig_hwc: np.ndarray, n_pert: int, seed: int) -> np.ndarray:
    """orig_hwc: [H,W,3] uint8 -> [n_pert,3,IMG_SIZE,IMG_SIZE] uint8 of bit-exact PIL
    views. When UPSCALE/DOWNSCALE, the original is LANCZOS-resized to IMG_SIZE *before* the
    perturbation pipeline runs, so the views live at the working resolution."""
    from PIL import Image
    random.seed(seed)
    torch.manual_seed(seed)
    img = Image.fromarray(orig_hwc)
    if _RESIZE:
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)   # best-quality LANCZOS resize first
    out = np.empty((n_pert, 3, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    for k in range(n_pert):
        aug = PRECOMPUTE_TF(img)                       # PIL RGB image
        out[k] = np.asarray(aug, dtype=np.uint8).transpose(2, 0, 1)
    return out


# Parallel precompute: workers write disjoint rows of the memmap directly.
_G = {}


def _init_worker(raw, train_idx, mmap_path, shape):
    _G["raw"] = raw
    _G["tidx"] = train_idx
    _G["mm"] = np.memmap(mmap_path, dtype=np.uint8, mode="r+", shape=shape)


def _precompute_one(j: int) -> int:
    orig = _G["raw"][_G["tidx"][j]]                    # [H,W,3] uint8 (32 CIFAR/SVHN, 64 TinyIN)
    _G["mm"][j] = pil_views(orig, NUM_PERTURB, seed=SEED * 1_000_003 + j)
    return j


# ---------------------------------------------------------------------------- #
# Data: precompute (or load) the view cache; keep originals for ground truth
# ---------------------------------------------------------------------------- #
def _cache_meta():
    m = {
        "num_perturb": NUM_PERTURB, "img_size": IMG_SIZE, "seed": SEED,
        "train_frac": TRAIN_FRAC, "train_subset": TRAIN_SUBSET, "version": 1,
        "upscale": UPSCALE, "upscale_size": UPSCALE_SIZE,
    }
    if DOWNSCALE:   # kept conditional so existing (non-downscale) caches stay valid
        m["downscale"], m["downscale_size"], m["source"] = True, DOWNSCALE_SIZE, "tiny_imagenet"
    if SVHN:        # conditional too, so plain-CIFAR-100 caches keep validating unchanged
        m["svhn"], m["source"] = True, "svhn"
    if STRONG_AUG:  # conditional so existing weak-aug caches keep validating unchanged
        m["strong_aug"] = True
    return m


def origs_to_tensor(hwc_batch: np.ndarray, device) -> torch.Tensor:
    """[K,H,W,3] uint8 originals -> [K,3,IMG_SIZE,IMG_SIZE] uint8 tensor on device.
    LANCZOS-resized to IMG_SIZE when UPSCALE/DOWNSCALE, so the ground truth matches the
    resolution the perturbed views were generated at."""
    if _RESIZE:
        hwc_batch = np.stack([resize_orig(o) for o in hwc_batch])
    return torch.from_numpy(np.ascontiguousarray(hwc_batch)).permute(0, 3, 1, 2).contiguous().to(device)


def _load_source_raw() -> np.ndarray:
    """Source originals as one [N,H,W,3] uint8 array. Plain/UPSCALE -> CIFAR-100 (.data,
    32px). SVHN -> SVHN train split, native 32px (source-independent control). DOWNSCALE ->
    Tiny ImageNet train loaded from ImageFolder into a 64px array, cached to disk as .npy so
    the 100k JPEGs are only read once (order is deterministic: ImageFolder sorts classes and
    files, and the .npy freezes it for the seeded split)."""
    if SVHN:
        # torchvision SVHN stores .data as [N,3,32,32] (CHW); transpose to the [N,H,W,3]
        # HWC convention every other source and downstream consumer here assumes.
        data = torchvision.datasets.SVHN(root=DATA_DIR, split="train", download=True).data
        return np.ascontiguousarray(data.transpose(0, 2, 3, 1))   # [N,32,32,3] uint8
    if not DOWNSCALE:
        return torchvision.datasets.CIFAR100(root=DATA_DIR, train=True, download=True).data
    from PIL import Image
    os.makedirs(CACHE_DIR, exist_ok=True)
    raw_cache = os.path.join(CACHE_DIR, "tiny_imagenet_train_raw.npy")
    if os.path.exists(raw_cache):
        return np.load(raw_cache)
    root = os.path.join(DATA_DIR, "tiny-imagenet-200", "train")
    samples = torchvision.datasets.ImageFolder(root).samples   # [(path, class), ...] sorted
    print(f"loading {len(samples)} Tiny ImageNet originals (64px, one-time) ...", flush=True)
    raw = np.empty((len(samples), 64, 64, 3), dtype=np.uint8)
    for i, (path, _) in enumerate(samples):
        with Image.open(path) as im:
            raw[i] = np.array(im.convert("RGB"), dtype=np.uint8)
    np.save(raw_cache, raw)
    return raw


def build_view_cache(device):
    os.makedirs(CACHE_DIR, exist_ok=True)
    raw = _load_source_raw()                           # [N,H,W,3] uint8 (CIFAR-100 or Tiny ImageNet)
    n = raw.shape[0]
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(n, generator=g).numpy()
    n_train = int(round(TRAIN_FRAC * n))
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    if TRAIN_SUBSET is not None:
        train_idx = train_idx[:TRAIN_SUBSET]
    n_train = len(train_idx)

    mmap_path = os.path.join(CACHE_DIR, f"train_views{_RES_TAG}{_AUG_TAG}.u8")
    meta_path = os.path.join(CACHE_DIR, f"train_views{_RES_TAG}{_AUG_TAG}.json")
    shape = (n_train, NUM_PERTURB, 3, IMG_SIZE, IMG_SIZE)
    meta = _cache_meta(); meta["n_train"] = n_train

    have = (not FORCE_PRECOMPUTE and os.path.exists(mmap_path) and os.path.exists(meta_path)
            and json.load(open(meta_path)) == meta
            and os.path.getsize(mmap_path) == int(np.prod(shape)))
    if have:
        print(f"using cached views: {mmap_path} ({np.prod(shape)/1e9:.1f} GB)")
    else:
        print(f"precomputing {n_train}x{NUM_PERTURB} PIL views "
              f"({np.prod(shape)/1e9:.1f} GB) on {PRECOMPUTE_WORKERS} workers...")
        mm = np.memmap(mmap_path, dtype=np.uint8, mode="w+", shape=shape)
        del mm                                         # workers reopen in r+ mode
        t0 = time.time()
        with Pool(PRECOMPUTE_WORKERS, initializer=_init_worker,
                  initargs=(raw, train_idx, mmap_path, shape)) as pool:
            done = 0
            for _ in pool.imap_unordered(_precompute_one, range(n_train), chunksize=16):
                done += 1
                if done % 2000 == 0 or done == n_train:
                    print(f"  {done}/{n_train}  ({time.time()-t0:.0f}s)", flush=True)
        json.dump(meta, open(meta_path, "w"))
        print(f"precompute done in {time.time()-t0:.0f}s")

    train_views = np.memmap(mmap_path, dtype=np.uint8, mode="r", shape=shape)
    train_orig = origs_to_tensor(raw[train_idx], device)

    # Test set is tiny (~50 originals) -> precompute its views in-process onto GPU.
    test_orig_np = raw[test_idx]
    test_views = np.stack([pil_views(test_orig_np[i], NUM_PERTURB, seed=SEED + 7 + i)
                           for i in range(len(test_idx))])
    test_views = torch.from_numpy(test_views).to(device)              # [nt,100,3,IMG,IMG] u8
    test_orig = origs_to_tensor(test_orig_np, device)
    return train_views, train_orig, test_views, test_orig


# ---------------------------------------------------------------------------- #
# GPU-side normalize + tile (identical math to ToTensor()+Normalize())
# ---------------------------------------------------------------------------- #
class GridMaker(nn.Module):
    def __init__(self, grid=GRID, img=IMG_SIZE):
        super().__init__()
        self.grid, self.img = grid, img
        self.register_buffer("mean", torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CIFAR_STD).view(1, 3, 1, 1))

    def norm(self, x_u8):
        return (x_u8.float() / 255.0 - self.mean) / self.std

    def leak_views(self, v):
        """Map clean normalized views [B,N,3,32,32] into the real-fragment input
        space: norm(to_unit(view)) (+ optional leak noise / bin contamination).

        Mirrors attacks.Loki.reconstruct (Eq. 9) followed by infer_fragments'
        to_unit+renorm. The Eq. 9 max-abs normalize is omitted because to_unit is
        scale-invariant and cancels it; leak noise is injected *before* to_unit
        (where it does not cancel) to emulate an imperfect real-training gradient."""
        if LEAK_CONTAM_P > 0:                              # bin contamination: blend a decoy
            b, n = v.shape[:2]
            flat = v.reshape(b * n, *v.shape[2:])
            perm = torch.randperm(flat.shape[0], device=v.device)
            mask = (torch.rand(flat.shape[0], device=v.device) < LEAK_CONTAM_P).view(-1, 1, 1, 1)
            flat = torch.where(mask, flat + LEAK_CONTAM_A * flat[perm], flat)
            v = flat.reshape(b, n, *v.shape[2:])
        if LEAK_NOISE > 0:                                 # imperfect real-training leak
            v = v + LEAK_NOISE * torch.randn_like(v)
        lo = v.amin(dim=(-3, -2, -1), keepdim=True)        # to_unit (infer_fragments mapping)
        hi = v.amax(dim=(-3, -2, -1), keepdim=True)
        u = (v - lo) / (hi - lo + 1e-8)
        return (u - self.mean.unsqueeze(1)) / self.std.unsqueeze(1)

    def norm_views(self, views_u8):
        """views_u8 [B,N,3,32,32] uint8 -> normalized float (per view).
        When LEAK_VIEWS, the views are remapped into the real-fragment input space."""
        v = (views_u8.float() / 255.0 - self.mean.unsqueeze(1)) / self.std.unsqueeze(1)
        return self.leak_views(v) if LEAK_VIEWS else v

    def grids(self, views_u8):
        """views_u8 [B,N,3,32,32] uint8 -> normalized grid [B,3,grid*32,grid*32]."""
        b = views_u8.shape[0]
        v = (views_u8.float() / 255.0 - self.mean.unsqueeze(1)) / self.std.unsqueeze(1)
        g, im = self.grid, self.img
        return (v.view(b, g, g, 3, im, im)
                 .permute(0, 3, 1, 4, 2, 5)
                 .reshape(b, 3, g * im, g * im).contiguous())


def denormalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR_MEAN, device=x.device).view(-1, 1, 1)
    std = torch.tensor(CIFAR_STD, device=x.device).view(-1, 1, 1)
    return (x * std + mean).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------- #
# Differentiable windowed SSIM (Gaussian window) -> the loss
# ---------------------------------------------------------------------------- #
def _gaussian_window(size, sigma, device):
    coords = torch.arange(size, device=device).float() - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


@torch.autocast("cuda", enabled=False)
def ssim(img1, img2, window_size=11, sigma=1.5, data_range=1.0):
    img1, img2 = img1.float(), img2.float()
    c = img1.shape[1]
    _1d = _gaussian_window(window_size, sigma, img1.device).to(img1.dtype)
    window = (_1d[:, None] @ _1d[None, :]).expand(c, 1, window_size, window_size).contiguous()
    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=c)
    mu2 = F.conv2d(img2, window, padding=pad, groups=c)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = F.conv2d(img1 * img1, window, padding=pad, groups=c) - mu1_sq
    s2 = F.conv2d(img2 * img2, window, padding=pad, groups=c) - mu2_sq
    s12 = F.conv2d(img1 * img2, window, padding=pad, groups=c) - mu1_mu2
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    m = ((2 * mu1_mu2 + c1) * (2 * s12 + c2)) / ((mu1_sq + mu2_sq + c1) * (s1 + s2 + c2))
    return m.mean()


def ssim_loss(pred_norm, gt_norm):
    return 1.0 - ssim(denormalize(pred_norm.float()), denormalize(gt_norm.float()), data_range=1.0)


# Frozen VGG16 perceptual loss: matching intermediate features (edges/textures)
# instead of raw pixels pulls the decoder off the blurry conditional mean.
_VGG = {}
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)
_VGG_TAPS = (3, 8, 15)        # relu1_2, relu2_2, relu3_3


def _vgg_extractor(device):
    if device not in _VGG:                             # cache PER device (dual-GPU replica needs its own)
        from torchvision.models import VGG16_Weights, vgg16
        feats = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:max(_VGG_TAPS) + 1].eval()
        for p in feats.parameters():
            p.requires_grad_(False)
        _VGG[device] = (feats.to(device),
                        torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1),
                        torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1))
    return _VGG[device]


@torch.autocast("cuda", enabled=False)
def vgg_perceptual(pred01, gt01):
    """pred01, gt01: images in [0,1]. L1 over VGG features (gt branch detached)."""
    feats, mean, std = _vgg_extractor(pred01.device)
    x = (pred01.float() - mean) / std
    y = (gt01.float() - mean) / std
    loss = 0.0
    for i, layer in enumerate(feats):
        x, y = layer(x), layer(y)
        if i in _VGG_TAPS:
            loss = loss + F.l1_loss(x, y.detach())
    return loss


# LPIPS perceptual distance (Zhang et al. 2018): calibrated linear weights on top of
# deep features -- a proper LEARNED perceptual metric, unlike the raw-VGG-feature L1
# above. Loaded lazily and frozen (we never train its weights); the same net compare_cid0
# scores with. normalize=True lets it take [0,1] inputs (it rescales to [-1,1] itself).
_LPIPS = {}


def _lpips_model(device):
    if device not in _LPIPS:                           # cache PER device (dual-GPU replica needs its own)
        import lpips as _lpips_pkg
        net = _lpips_pkg.LPIPS(net=LPIPS_NET, verbose=False).to(device).eval()
        for p in net.parameters():
            p.requires_grad_(False)                    # freeze: metric, not a trained head
        _LPIPS[device] = net
    return _LPIPS[device]


@torch.autocast("cuda", enabled=False)
def lpips_perceptual(pred01, gt01):
    """pred01, gt01 in [0,1]. Mean LPIPS distance (lower = perceptually closer); gt
    branch detached. Runs fp32 (autocast off) like the SSIM/VGG terms. Freezing the net
    stops its weights updating but still lets gradients flow back into pred01."""
    net = _lpips_model(pred01.device)
    return net(pred01.float(), gt01.float().detach(), normalize=True).mean()


def recon_loss(pred_norm, gt_norm, ssim_w=0.84, vgg_w=PERCEPTUAL_W):
    """Structure (1-SSIM) + low-freq/colour fidelity (L1) + optional VGG perceptual.
    Pure 1-SSIM leaves global colour drift; L1 pins it down; the VGG term rewards
    matching edges/textures, which counters the blurry mean-seeking of pixel losses.
    LPIPS_LOSS instead trains on the LPIPS distance (+ a small L1 anchor for colour)."""
    p = denormalize(pred_norm.float())
    g = denormalize(gt_norm.float())
    if LPIPS_LOSS:
        return LPIPS_W * lpips_perceptual(p, g) + LPIPS_L1_W * F.l1_loss(p, g)
    loss = ssim_w * (1.0 - ssim(p, g, data_range=1.0)) + (1.0 - ssim_w) * F.l1_loss(p, g)
    if vgg_w > 0:
        loss = loss + vgg_w * vgg_perceptual(p, g)
    return loss


# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #
def tile_grid(v, grid=GRID, img=IMG_SIZE):
    """[B,N,3,32,32] normalized views -> [B,3,grid*32,grid*32] tiled image."""
    b = v.shape[0]
    return (v.view(b, grid, grid, 3, img, img)
             .permute(0, 3, 1, 4, 2, 5)
             .reshape(b, 3, grid * img, grid * img).contiguous())


class ReconCNN(nn.Module):
    """Legacy baseline: tile the views into a grid and run a plain CNN over it."""
    def __init__(self, grid=GRID, img_size=IMG_SIZE):
        super().__init__()
        self.img_size = img_size

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        self.features = nn.Sequential(
            block(3, 64), block(64, 128), block(128, 256),
            block(256, 256), block(256, 512), block(512, 512))
        side = grid * img_size
        for _ in range(6):
            side = math.ceil(side / 2)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * side * side, 2048), nn.ReLU(inplace=True),
            nn.Linear(2048, 3 * img_size * img_size))

    def forward(self, views):
        x = tile_grid(views).to(memory_format=torch.channels_last)
        return self.head(self.features(x)).view(-1, 3, self.img_size, self.img_size)


class SetReconNet(nn.Module):
    """Treat the N views as an unordered SET.

    A shared CNN encodes each 32x32 view to a vector; the set is collapsed
    permutation-invariantly (attention-weighted + mean pool) so view order /
    count don't matter and useless views (heavy crops, grayscale) can be
    down-weighted; an upsampling conv decoder synthesises the canonical 32x32
    original from the aggregated code -- no FC-to-pixels bottleneck.
    """
    def __init__(self, img_size=IMG_SIZE, feat=512):
        super().__init__()
        self.img_size, self.feat = img_size, feat

        def enc_block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        # per-view encoder: 32 -> 16 -> 8 -> 4 -> 2 -> global pool
        self.enc = nn.Sequential(
            enc_block(3, 64), enc_block(64, 128), enc_block(128, 256),
            enc_block(256, feat), nn.AdaptiveAvgPool2d(1))
        # attention scorer over the set (per-view scalar logit)
        self.attn = nn.Sequential(nn.Linear(feat, feat), nn.Tanh(), nn.Linear(feat, 1))
        # seed a 4x4 feature map from the aggregated code (attn ++ mean)
        self.fc = nn.Linear(feat * 2, feat * 4 * 4)

        def up_block(cin, cout):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        # decoder: 4 -> 8 -> 16 -> 32
        self.dec = nn.Sequential(
            up_block(feat, 256), up_block(256, 128), up_block(128, 64),
            nn.Conv2d(64, 3, 3, padding=1))

    def forward(self, views):                              # [B,N,3,32,32] normalized
        b, n = views.shape[:2]
        x = views.reshape(b * n, *views.shape[2:]).to(memory_format=torch.channels_last)
        f = self.enc(x).flatten(1).view(b, n, self.feat)   # [B,N,feat]
        w = self.attn(f).softmax(dim=1)                    # [B,N,1]
        z = torch.cat([(w * f).sum(1), f.mean(1)], dim=1)  # [B,2*feat]
        seed = self.fc(z).view(b, self.feat, 4, 4)
        return self.dec(seed)                              # [B,3,32,32]


class SetTransformer(nn.Module):
    """Each of the N views is a TOKEN.

    A shared CNN stem embeds every 32x32 view into a token; a Transformer
    encoder runs self-attention *across the view tokens* so each view is
    refined by all the others (the cross-view triangulation a set encoder
    can't do). Deliberately NO positional encoding on the tokens -> the model
    is permutation-equivariant over the set, so view order/count are irrelevant.
    A small grid of learned query tokens then cross-attends to the refined
    views (DETR-style) to seed a conv decoder -- a grid of distinct codes
    rather than one pooled vector, so different output regions can draw on
    different views.
    """
    def __init__(self, img_size=IMG_SIZE, dim=512, heads=8, enc_layers=6,
                 dec_layers=4, seed_grid=16, dropout=DROPOUT):
        super().__init__()
        assert dim % heads == 0, f"dim ({dim}) must be divisible by heads ({heads})"
        ratio = img_size // seed_grid
        assert seed_grid <= img_size and img_size % seed_grid == 0 and (ratio & (ratio - 1)) == 0, \
            f"img_size/seed_grid must be a power of 2 (got {img_size}/{seed_grid})"
        self.img_size, self.dim, self.seed_grid = img_size, dim, seed_grid

        def enc_block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        # per-view stem: 32 -> 16 -> 8 -> 4 -> 2 -> global pool -> token
        self.stem = nn.Sequential(
            enc_block(3, 64), enc_block(64, 128), enc_block(128, 256),
            enc_block(256, dim), nn.AdaptiveAvgPool2d(1))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)

        # learned spatial query tokens that read out a seed_grid x seed_grid map
        self.queries = nn.Parameter(torch.randn(seed_grid * seed_grid, dim) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_layers)

        def up_block(cin, cout):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        # upsample seed_grid -> img_size, halving channels each step (floor 64)
        n_up = int(math.log2(ratio))
        chans, c = [dim], dim
        for _ in range(n_up):
            c = max(64, c // 2)
            chans.append(c)
        self.to_img = nn.Sequential(
            *[up_block(chans[i], chans[i + 1]) for i in range(n_up)],
            nn.Conv2d(chans[-1], 3, 3, padding=1))

    def forward(self, views):                              # [B,N,3,32,32] normalized
        b, n = views.shape[:2]
        x = views.reshape(b * n, *views.shape[2:]).to(memory_format=torch.channels_last)
        tok = self.stem(x).flatten(1).view(b, n, self.dim)  # [B,N,dim] set tokens (no pos emb)
        mem = self.encoder(tok)                             # [B,N,dim] cross-view refined
        q = self.queries.unsqueeze(0).expand(b, -1, -1)     # [B,M,dim]
        dec = self.decoder(q, mem)                          # [B,M,dim]
        g = self.seed_grid
        seed = dec.transpose(1, 2).reshape(b, self.dim, g, g)
        return self.to_img(seed)                            # [B,3,32,32]


def make_optimizer(model, lr=LR, weight_decay=WEIGHT_DECAY):
    """AdamW with weight decay applied only to >=2D weights (not norms/biases)."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or n.endswith(".bias") else decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def build_model(name=MODEL):
    if name == "settr":
        return SetTransformer()
    if name == "set":
        return SetReconNet()
    if name == "cnn":
        return ReconCNN()
    raise ValueError(f"unknown model '{name}'")


# ---------------------------------------------------------------------------- #
# Train / eval
# ---------------------------------------------------------------------------- #
def iter_batches(n, batch, shuffle=True):
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    stop = n - (batch if shuffle else 0)
    for i in range(0, max(stop, 1) if not shuffle else stop, batch):
        yield idx[i:i + batch]


def fetch_views(train_views, sel_cpu, device):
    """Gather a batch of rows from the (CPU) memmap and move to GPU as uint8."""
    arr = train_views[sel_cpu.numpy()]                 # [B,N,3,32,32] uint8 (RAM)
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device, non_blocking=True)


def call_model(model, views, compiled):
    """Forward, marking the view-count axis (dim 1) dynamic so torch.compile builds
    ONE graph for the whole [VIEWS_MIN, VIEWS_MAX] spread instead of recompiling per k.
    The batch and spatial dims are static (full batches, fixed IMG_SIZE), so only dim 1
    is flagged. No-op semantics when not compiled."""
    if compiled:
        torch._dynamo.mark_dynamic(views, 1)
    return model(views)


def make_epoch_plan(n_train, batch, n1):
    """One epoch's worth of (sel, k, pick) steps, drawn from the SAME RNGs the old inline
    loop used so training is unchanged: k log-uniform in [VIEWS_MIN, VIEWS_MAX] (python
    `random`), pick a random view subset (torch). For DUAL_GPU the first `n1` originals of
    each sel go to the weak GPU1, the rest to GPU0 -- decided here so the prefetcher can
    ship the two shards straight to their devices."""
    plan = []
    for sel in iter_batches(n_train, batch, shuffle=True):
        hi = min(VIEWS_MAX, NUM_PERTURB)
        lo = min(max(1, VIEWS_MIN), hi)
        k = int(round(math.exp(random.uniform(math.log(lo), math.log(hi)))))
        k = max(lo, min(k, hi))
        pick = torch.randperm(NUM_PERTURB)[:k].numpy()
        plan.append((sel, pick))
    return plan


def prepare_batch(train_views, train_orig, sel, pick, dev0, dev1, n1):
    """Gather the `pick` views for originals `sel` from the CPU memmap and ship them to
    the GPU(s) as uint8. Single-GPU (dev1 None) -> (v0, gt0, None, None). Dual -> the
    first n1 originals go to the weak GPU1, the rest to dev0; gt comes from `train_orig`
    (already on dev0), and gt1 is copied dev0->dev1."""
    arr = np.ascontiguousarray(train_views[sel.numpy()][:, pick])    # [B,k,3,H,W] u8, subsampled
    t = torch.from_numpy(arr)
    if dev1 is None:
        v0 = t.pin_memory().to(dev0, non_blocking=True)
        return v0, train_orig[sel.to(dev0)], None, None
    v1 = t[:n1].contiguous().pin_memory().to(dev1, non_blocking=True)
    v0 = t[n1:].contiguous().pin_memory().to(dev0, non_blocking=True)
    gt0 = train_orig[sel[n1:].to(dev0)]
    gt1 = train_orig[sel[:n1].to(dev0)].to(dev1)
    return v0, gt0, v1, gt1


class CudaPrefetcher:
    """Runs prepare_batch for the NEXT step in a background thread so the CPU memmap
    gather + host->device copy overlaps the current step's GPU compute. The loader was
    fully synchronous before -- ~79 MB gathered per step with the GPU idle. Iterating
    yields the same (v0, gt0, v1, gt1) tuples prepare_batch returns."""
    def __init__(self, train_views, train_orig, plan, dev0, dev1, n1):
        from concurrent.futures import ThreadPoolExecutor
        self.tv, self.torig, self.plan = train_views, train_orig, plan
        self.dev0, self.dev1, self.n1 = dev0, dev1, n1
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.i = 0
        self.fut = self.pool.submit(self._load, 0) if plan else None

    def _load(self, idx):
        sel, pick = self.plan[idx]
        return prepare_batch(self.tv, self.torig, sel, pick, self.dev0, self.dev1, self.n1)

    def __iter__(self):
        return self

    def __next__(self):
        if self.fut is None:
            raise StopIteration
        out = self.fut.result()
        self.i += 1
        self.fut = self.pool.submit(self._load, self.i) if self.i < len(self.plan) else None
        return out

    def close(self):
        self.pool.shutdown(wait=False)


@torch.no_grad()
def evaluate(model, gm, test_views, test_orig, device):
    """Mean test SSIM (always) plus mean test LPIPS when LPIPS_LOSS is on. Returns
    (ssim, lpips); lpips is None unless LPIPS_LOSS, so checkpointing can select on the
    objective it actually trained (lowest LPIPS) rather than SSIM."""
    model.eval()
    ssim_sum, lpips_sum, seen = 0.0, 0.0, 0
    n = test_views.shape[0]
    for i in range(0, n, BATCH_SIZE):
        vb = test_views[i:i + BATCH_SIZE]
        views = gm.norm_views(vb)
        gt = gm.norm(test_orig[i:i + BATCH_SIZE])
        with torch.autocast("cuda", enabled=AMP and device.type == "cuda"):
            pred = model(views)
        p01, g01 = denormalize(pred.float()), denormalize(gt)
        ssim_sum += float(ssim(p01, g01, data_range=1.0)) * vb.shape[0]
        if LPIPS_LOSS:
            lpips_sum += float(lpips_perceptual(p01, g01)) * vb.shape[0]
        seen += vb.shape[0]
    seen = max(1, seen)
    return ssim_sum / seen, (lpips_sum / seen if LPIPS_LOSS else None)


@torch.no_grad()
def save_preview(model, gm, test_views, test_orig, device, path, max_show=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    model.eval()
    n = min(max_show, test_views.shape[0])
    views = gm.norm_views(test_views[:n])
    gt = gm.norm(test_orig[:n])
    with torch.autocast("cuda", enabled=AMP and device.type == "cuda"):
        pred = model(views)
    gt_u, pr_u = denormalize(gt).cpu(), denormalize(pred.float()).cpu()
    fig, axes = plt.subplots(2, n, figsize=(n * 1.3, 2.8))
    axes = np.atleast_2d(axes).reshape(2, n)
    for k in range(n):
        axes[0][k].imshow(gt_u[k].permute(1, 2, 0).numpy()); axes[0][k].axis("off")
        axes[1][k].imshow(pr_u[k].permute(1, 2, 0).numpy()); axes[1][k].axis("off")
    fig.suptitle("top: ground truth   bottom: reconstruction")
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)


def train_forward(tmodel, gm, v_u8, gt_u8, dev):
    """norm views + originals, autocast forward through (maybe compiled) tmodel, recon_loss.
    Returns the shard's mean loss on `dev`. Shared by the primary and the dual-GPU replica."""
    views = gm.norm_views(v_u8)
    gt = gm.norm(gt_u8)
    with torch.autocast("cuda", enabled=AMP and dev.type == "cuda"):
        pred = call_model(tmodel, views, COMPILE)
        return recon_loss(pred, gt)


def main():
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    device = get_device()
    dev0 = torch.device("cuda:0") if device.type == "cuda" else device
    os.makedirs(OUT_DIR, exist_ok=True)

    train_views, train_orig, test_views, test_orig = build_view_cache(dev0)
    n_train = train_views.shape[0]
    print(f"train originals: {n_train} | test originals: {test_views.shape[0]} | "
          f"{NUM_PERTURB} views each, grid {GRID}x{GRID}")

    gm = GridMaker().to(dev0)
    model = build_model(MODEL).to(dev0).to(memory_format=torch.channels_last)
    opt = make_optimizer(model, lr=LR, weight_decay=WEIGHT_DECAY)
    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, EPOCHS - WARMUP_EPOCHS))
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], milestones=[WARMUP_EPOCHS])
    scaler = torch.amp.GradScaler("cuda", enabled=AMP and dev0.type == "cuda")

    # --- dual-GPU replica: a second copy of the model on cuda:1 that runs the small shard.
    # It has NO optimizer -- the single opt on cuda:0 owns the update, and after each step
    # the fresh weights are copied back here (see the loop). Only `model` is checkpointed.
    dual = DUAL_GPU and dev0.type == "cuda" and torch.cuda.device_count() >= 2
    if dual:
        dev1 = torch.device("cuda:1")
        gm1 = GridMaker().to(dev1)
        model1 = build_model(MODEL).to(dev1).to(memory_format=torch.channels_last)
        with torch.no_grad():                              # replica starts bit-identical to primary
            for p1, p0 in zip(model1.parameters(), model.parameters()):
                p1.copy_(p0.to(dev1))
            for b1, b0 in zip(model1.buffers(), model.buffers()):
                b1.copy_(b0.to(dev1))
        n1 = min(max(1, round(GPU1_FRAC * BATCH_SIZE)), BATCH_SIZE - 1)   # weak-GPU shard size
        n0 = BATCH_SIZE - n1
    else:
        dev1, gm1, model1, n1, n0 = None, None, None, 0, BATCH_SIZE

    train_model = torch.compile(model) if COMPILE else model
    train_model1 = torch.compile(model1) if (COMPILE and dual) else model1

    print(f"model: {MODEL} | params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M | "
          f"device: {dev0} | amp: {AMP} | batch: {BATCH_SIZE} | compile: {COMPILE} | prefetch: {PREFETCH}")
    if dual:
        print(f"DUAL_GPU on -> split {n0} originals on {dev0} + {n1} on {dev1} "
              f"(GPU1_FRAC={GPU1_FRAC}); size-weighted grad combine, single optimizer on {dev0}")
    if LEAK_VIEWS:
        print(f"LEAK_VIEWS on -> training on real-fragment input space "
              f"(noise={LEAK_NOISE}, contam_p={LEAK_CONTAM_P}) -> {CKPT_NAME}")

    random.seed(SEED)            # reproducible per-step view-count sampling (make_epoch_plan)
    best = float("-inf")         # works for both "higher SSIM" and "lower LPIPS" (stored negated)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        if dual:
            model1.train()
        t0, running, seen = time.time(), 0.0, 0
        plan = make_epoch_plan(n_train, BATCH_SIZE, n1)
        loader = (CudaPrefetcher(train_views, train_orig, plan, dev0, dev1, n1) if PREFETCH else
                  (prepare_batch(train_views, train_orig, sel, pick, dev0, dev1, n1) for sel, pick in plan))
        for v0, gt0, v1, gt1 in loader:
            opt.zero_grad(set_to_none=True)
            if dual:
                model1.zero_grad(set_to_none=True)
            loss0 = train_forward(train_model, gm, v0, gt0, dev0)
            if not torch.isfinite(loss0):
                continue                                   # skip a poisoned batch, keep weights
            if dual:
                loss1 = train_forward(train_model1, gm1, v1, gt1, dev1)
                if not torch.isfinite(loss1):
                    continue
                s = scaler.get_scale()                     # same loss-scale S on both graphs
                scaler.scale(loss0).backward()             # model.grad  = S * dloss0/dw  (mean over n0)
                (loss1 * s).backward()                     # model1.grad = S * dloss1/dw  (mean over n1)
                # Size-weighted combine == the exact gradient of one B-sized batch:
                #   (n0*g0 + n1*g1)/B = S/B * (sum_{S0} grad l_i + sum_{S1} grad l_i)
                with torch.no_grad():
                    for p0_, p1_ in zip(model.parameters(), model1.parameters()):
                        if p0_.grad is None and p1_.grad is None:
                            continue
                        g0 = p0_.grad if p0_.grad is not None else torch.zeros_like(p0_)
                        g1 = p1_.grad.to(dev0) if p1_.grad is not None else 0.0
                        p0_.grad = (n0 * g0 + n1 * g1) / BATCH_SIZE
                step_loss = (n0 * loss0.detach().item() + n1 * loss1.detach().item()) / BATCH_SIZE
            else:
                scaler.scale(loss0).backward()
                step_loss = loss0.detach().item()
            scaler.unscale_(opt)                           # unscale before clipping
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            if not torch.isfinite(gnorm):
                opt.zero_grad(set_to_none=True)
                scaler.update()                            # drop this step, let scaler retune
                continue                                   # never apply a non-finite-grad update
            scaler.step(opt)
            scaler.update()
            if dual:                                       # broadcast fresh weights to the replica
                with torch.no_grad():
                    for p0_, p1_ in zip(model.parameters(), model1.parameters()):
                        p1_.copy_(p0_, non_blocking=True)
            running += step_loss * BATCH_SIZE
            seen += BATCH_SIZE
        if PREFETCH:
            loader.close()
        sched.step()
        train_loss = running / max(1, seen)
        test_ssim, test_lpips = evaluate(model, gm, test_views, test_orig, dev0)
        loss_name = "LPIPS" if LPIPS_LOSS else "1-SSIM"
        line = (f"epoch {epoch:3d} | lr {sched.get_last_lr()[0]:.2e} | "
                f"train {loss_name} {train_loss:.4f} | test SSIM {test_ssim:.4f}")
        if LPIPS_LOSS:
            line += f" | test LPIPS {test_lpips:.4f}"
        print(line + f" | {time.time()-t0:.1f}s", flush=True)

        save_preview(model, gm, test_views, test_orig, dev0,
                     os.path.join(OUT_DIR, f"preview_epoch{epoch:03d}.png"))
        # Select the checkpoint on the trained objective: lowest LPIPS (stored as its
        # negative so "> best" still means "better") in LPIPS mode, else highest SSIM.
        score = -test_lpips if LPIPS_LOSS else test_ssim
        if score > best:
            best = score
            torch.save(model.state_dict(), os.path.join(OUT_DIR, CKPT_NAME))

    best_str = f"LPIPS {-best:.4f}" if LPIPS_LOSS else f"SSIM {best:.4f}"
    print(f"done. best test {best_str} | checkpoints + previews in {OUT_DIR}/")


if __name__ == "__main__":
    main()

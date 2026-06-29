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
IMG_SIZE      = 32
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
AMP           = True         # mixed precision
PRECOMPUTE_WORKERS = 24
FORCE_PRECOMPUTE = False     # ignore cache and regenerate the views
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
LEAK_VIEWS    = True         # train/eval on LOKI-leak-style views (match real fragments)
LEAK_NOISE    = 0.0           # additive noise std (model-input units) injected pre-to_unit
LEAK_CONTAM_P = 0.0           # per-view prob. of blending a decoy view (bin contamination)
LEAK_CONTAM_A = 0.5           # decoy blend weight when a view is contaminated
CKPT_NAME     = "best_leak.pt" if LEAK_VIEWS else "best.pt"   # don't clobber the clean ckpt

# CIFAR normalization (must match the federated pipeline exactly).
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

assert GRID * GRID == NUM_PERTURB, "NUM_PERTURB must be a perfect square == GRID*GRID"

# Exact client augmentation MINUS the final ToTensor/Normalize. We keep the output
# as a uint8 PIL image (same pixels the client feeds ToTensor) and do /255 +
# normalize on the GPU -- mathematically identical to ToTensor()+Normalize().
_COLOR_JITTER = T.ColorJitter(0.8, 0.8, 0.8, 0.2)
PRECOMPUTE_TF = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomApply([_COLOR_JITTER], p=0.8),
    T.RandomGrayscale(p=0.2),
])
loss_func = nn.MSELoss()

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------- #
# PIL view generation (used by both the parallel precompute and the test set)
# ---------------------------------------------------------------------------- #
def pil_views(orig_hwc: np.ndarray, n_pert: int, seed: int) -> np.ndarray:
    """orig_hwc: [32,32,3] uint8 -> [n_pert,3,32,32] uint8 of bit-exact PIL views."""
    from PIL import Image
    random.seed(seed)
    torch.manual_seed(seed)
    img = Image.fromarray(orig_hwc)
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
    orig = _G["raw"][_G["tidx"][j]]                    # [32,32,3] uint8
    _G["mm"][j] = pil_views(orig, NUM_PERTURB, seed=SEED * 1_000_003 + j)
    return j


# ---------------------------------------------------------------------------- #
# Data: precompute (or load) the view cache; keep originals for ground truth
# ---------------------------------------------------------------------------- #
def _cache_meta():
    return {
        "num_perturb": NUM_PERTURB, "img_size": IMG_SIZE, "seed": SEED,
        "train_frac": TRAIN_FRAC, "train_subset": TRAIN_SUBSET, "version": 1,
    }


def build_view_cache(device):
    os.makedirs(CACHE_DIR, exist_ok=True)
    base = torchvision.datasets.CIFAR100(root=DATA_DIR, train=True, download=True)
    raw = base.data                                    # [50000,32,32,3] uint8
    n = raw.shape[0]
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(n, generator=g).numpy()
    n_train = int(round(TRAIN_FRAC * n))
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    if TRAIN_SUBSET is not None:
        train_idx = train_idx[:TRAIN_SUBSET]
    n_train = len(train_idx)

    mmap_path = os.path.join(CACHE_DIR, "train_views.u8")
    meta_path = os.path.join(CACHE_DIR, "train_views.json")
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
    train_orig = torch.from_numpy(raw[train_idx]).permute(0, 3, 1, 2).contiguous().to(device)

    # Test set is tiny (~50 originals) -> precompute its views in-process onto GPU.
    test_orig_np = raw[test_idx]
    test_views = np.stack([pil_views(test_orig_np[i], NUM_PERTURB, seed=SEED + 7 + i)
                           for i in range(len(test_idx))])
    test_views = torch.from_numpy(test_views).to(device)              # [nt,100,3,32,32] u8
    test_orig = torch.from_numpy(test_orig_np).permute(0, 3, 1, 2).contiguous().to(device)
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
    if "feats" not in _VGG:
        from torchvision.models import VGG16_Weights, vgg16
        feats = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:max(_VGG_TAPS) + 1].eval()
        for p in feats.parameters():
            p.requires_grad_(False)
        _VGG["feats"] = feats.to(device)
        _VGG["mean"] = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        _VGG["std"] = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return _VGG["feats"], _VGG["mean"], _VGG["std"]


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


def recon_loss(pred_norm, gt_norm, ssim_w=0.84, vgg_w=PERCEPTUAL_W):
    """Structure (1-SSIM) + low-freq/colour fidelity (L1) + optional VGG perceptual.
    Pure 1-SSIM leaves global colour drift; L1 pins it down; the VGG term rewards
    matching edges/textures, which counters the blurry mean-seeking of pixel losses."""
    p = denormalize(pred_norm.float())
    g = denormalize(gt_norm.float())
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


@torch.no_grad()
def evaluate(model, gm, test_views, test_orig, device):
    model.eval()
    total, seen = 0.0, 0
    n = test_views.shape[0]
    for i in range(0, n, BATCH_SIZE):
        vb = test_views[i:i + BATCH_SIZE]
        views = gm.norm_views(vb)
        gt = gm.norm(test_orig[i:i + BATCH_SIZE])
        with torch.autocast("cuda", enabled=AMP and device.type == "cuda"):
            pred = model(views)
        total += float(ssim(denormalize(pred.float()), denormalize(gt), data_range=1.0)) * vb.shape[0]
        seen += vb.shape[0]
    return total / max(1, seen)


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


def main():
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    device = get_device()
    os.makedirs(OUT_DIR, exist_ok=True)

    train_views, train_orig, test_views, test_orig = build_view_cache(device)
    n_train = train_views.shape[0]
    print(f"train originals: {n_train} | test originals: {test_views.shape[0]} | "
          f"{NUM_PERTURB} views each, grid {GRID}x{GRID}")

    gm = GridMaker().to(device)
    model = build_model(MODEL).to(device).to(memory_format=torch.channels_last)
    opt = make_optimizer(model, lr=LR, weight_decay=WEIGHT_DECAY)
    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, EPOCHS - WARMUP_EPOCHS))
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], milestones=[WARMUP_EPOCHS])
    scaler = torch.amp.GradScaler("cuda", enabled=AMP and device.type == "cuda")
    print(f"model: {MODEL} | params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M | "
          f"device: {device} | amp: {AMP} | batch: {BATCH_SIZE}")
    if LEAK_VIEWS:
        print(f"LEAK_VIEWS on -> training on real-fragment input space "
              f"(noise={LEAK_NOISE}, contam_p={LEAK_CONTAM_P}) -> {CKPT_NAME}")

    random.seed(SEED)            # reproducible per-step view-count sampling below
    best = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        for sel in iter_batches(n_train, BATCH_SIZE, shuffle=True):
            vb = fetch_views(train_views, sel, device)
            # Per-step random view count (log-uniform in [VIEWS_MIN, VIEWS_MAX]),
            # resampled every step so the model sees the full spread of token counts
            # it faces on real clusters rather than one fixed count. One count per
            # step (shared across the batch); the SetTransformer is count-invariant.
            navail = vb.shape[1]
            hi = min(VIEWS_MAX, navail)
            lo = min(max(1, VIEWS_MIN), hi)
            k = int(round(math.exp(random.uniform(math.log(lo), math.log(hi)))))
            k = max(lo, min(k, hi))
            if k < navail:
                pick = torch.randperm(navail, device=vb.device)[:k]
                vb = vb[:, pick]                           # random view subset, resampled each step
            views = gm.norm_views(vb)
            gt = gm.norm(train_orig[sel.to(device)])
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=AMP and device.type == "cuda"):
                pred = model(views)
                loss = recon_loss(pred, gt)
            if not torch.isfinite(loss):
                continue                                   # skip a poisoned batch, keep weights
            scaler.scale(loss).backward()
            scaler.unscale_(opt)                           # unscale before clipping
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            if not torch.isfinite(gnorm):
                opt.zero_grad(set_to_none=True)
                scaler.update()                            # drop this step, let scaler retune
                continue                                   # never apply a non-finite-grad update
            scaler.step(opt)
            scaler.update()
            running += float(loss) * len(sel)
            seen += len(sel)
        sched.step()
        train_loss = running / max(1, seen)
        test_ssim = evaluate(model, gm, test_views, test_orig, device)
        print(f"epoch {epoch:3d} | lr {sched.get_last_lr()[0]:.2e} | train 1-SSIM {train_loss:.4f} | "
              f"test SSIM {test_ssim:.4f} | {time.time()-t0:.1f}s", flush=True)

        save_preview(model, gm, test_views, test_orig, device,
                     os.path.join(OUT_DIR, f"preview_epoch{epoch:03d}.png"))
        if test_ssim > best:
            best = test_ssim
            torch.save(model.state_dict(), os.path.join(OUT_DIR, CKPT_NAME))

    print(f"done. best test SSIM {best:.4f} | checkpoints + previews in {OUT_DIR}/")


if __name__ == "__main__":
    main()

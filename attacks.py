"""
attacks.py — LOKI data-reconstruction attack against federated learning.

Faithful implementation of the malicious-server linear-layer-leakage attack from

    Zhao, Sharma, Elkordy, Ezzeldin, Avestimehr, Bagchi,
    "LOKI: Large-scale Data Reconstruction Attack against Federated Learning
     through Model Manipulation", IEEE S&P 2024.

The single public class `Loki` bundles every mechanism the paper describes:

  * the inserted attack module — one convolutional layer of *identity mapping
    sets* followed by two fully-connected layers, with input/output dimensions
    matching the image so the benign model is unchanged (Sec. 3.3, Fig. 2);
  * the identity mapping sets that push each client's image forward on a
    *disjoint* set of conv-output channels, so the FC weight gradients stay
    separable through secure aggregation (Sec. 3.4-3.5);
  * the first FC layer set up for "binning" linear-layer leakage — weights
    measuring average pixel intensity, biases as sorted cutoffs — using a
    double-bounded (Hardtanh) activation for FedAVG (Eq. 4) and the inter-bias
    parameter scaling of Eq. 5;
  * the convolutional scaling factor (CSF) of Eq. 10 that fights FP32 precision
    loss and repeat activations across local iterations (Sec. 3.6);
  * bias-free reconstruction from the (aggregated) weight gradient via Eq. 9,
    de-aggregated per identity mapping set (Eq. 8), tracing every reconstruction
    back to its owning client (Sec. 3.5, 4.2);
  * incremental learning of the per-client bias distribution across rounds
    (Sec. 4.1) and the parameter/sparsity accounting of Sec. 4.3.

Threat model (Sec. 3.1): a malicious server that manipulates the architecture
and parameters it sends to clients, then reconstructs from the returned update.

This file provides the reusable building blocks: `LokiModule` (the inserted
attack module, prepended to the encoder by `architectures.LokiEncoder`) and the
`Loki` class (arming the trap weights and reconstructing from an update). The
*in-the-loop* attack that actually runs in federation lives in
`server.FedEMAStrategyWithKnn`: each round it arms the trap module for the target
client only (model inconsistency, via `configure_fit`) and reconstructs that
client's images from its returned update (via `aggregate_fit`).

`simulate_client_update` and `attack` here are offline helpers — they stand in
for a client training the trap model so the reconstruction math can be validated
without a full Flower run. They are not used by the live server attack.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# CIFAR-10 normalization (matches dataloader.py) — used only to de-normalize
# reconstructions for display, never by the attack itself.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


# --------------------------------------------------------------------------- #
# Attack module (Fig. 2): conv identity mapping sets -> FC1 (leak) -> FC2.
# --------------------------------------------------------------------------- #
class LokiModule(nn.Module):
    """
    The module the malicious server prepends to the benign model.

    Layout (Sec. 3.3): a convolutional layer with `num_kernels` kernels holding
    the identity mapping sets, followed by FC1 (the leak layer whose weight
    gradient carries the data) and FC2 (the all-to-all layer that maps back to
    image dimensions so the benign model downstream is unaffected).

    The conv uses 3x3 kernels with padding 1 and stride 1, so each kernel is a
    spatial identity (a single non-zero "key value" `kv` at the centre tap) that
    copies one input channel to one output channel — i.e. it *pushes the image
    forward* without distortion. FC1 weights are constant (they measure average
    pixel intensity); FC2 has identical rows (Sec. 2.3) so every FC1 neuron sees
    the same downstream scaling, which is what keeps the binning identity valid.
    """

    def __init__(self, image_shape: Tuple[int, int, int], fc_size: int, num_kernels: int,
                 fedavg: bool = True):
        super().__init__()
        C, H, W = image_shape
        self.image_shape = (C, H, W)
        self.C, self.H, self.W = C, H, W
        self.fc_size = fc_size
        self.num_kernels = num_kernels

        self.conv_out_elems = num_kernels * H * W
        self.image_elems = C * H * W

        # Conv: keeps spatial size (k=3, pad=1, stride=1); no bias (identity map).
        self.conv = nn.Conv2d(C, num_kernels, kernel_size=3, stride=1, padding=1, bias=False)
        # FC1 is the leak layer: its weight gradient reconstructs the inputs.
        self.fc1 = nn.Linear(self.conv_out_elems, fc_size, bias=True)
        # FC2 maps back to image dimensions so the inserted module is shape-preserving.
        self.fc2 = nn.Linear(fc_size, self.image_elems, bias=False)

        # The leak is read solely from fc1's weight/bias update; conv (identity
        # map) and fc2 (constant residual back-projection) are re-armed by the
        # server every round, so any client update to them is discarded. Freeze
        # them: this drops their gradients *and* SGD momentum buffers (fc2 alone
        # is fc_size*C*H*W params — ~0.5 GB each in fp32), a large VRAM saving.
        # The fc1 leak is unaffected: autograd still flows *through* a frozen fc2
        # to fc1; the flag only stops fc2/conv from accumulating their own grad.
        self.conv.weight.requires_grad_(False)
        self.fc2.weight.requires_grad_(False)

        # Activation choice depends on the federated mode:
        #   * FedAVG: Hardtanh (Eq. 4) — double-bounded, non-zero gradient only on
        #     inputs in (0, 1). Combined with the Eq. 5 band-alignment scaling,
        #     each neuron fires for just its own brightness bin, giving the sparse
        #     activation FedAVG needs.
        #   * FedSGD: a one-sided ReLU — a neuron fires for every image brighter
        #     than its cutoff, so the consecutive-bin gradient difference (Eq. 3)
        #     cancels the shared brighter images and isolates one input exactly.
        self.activation = nn.Hardtanh(0.0, 1.0) if fedavg else nn.ReLU()

    def kernel_center(self) -> Tuple[int, int]:
        """Centre tap of the conv kernel — where the identity key value goes."""
        kh, kw = self.conv.kernel_size
        return kh // 2, kw // 2

    def forward(self, x: torch.Tensor, residual: bool = True) -> torch.Tensor:
        conv_out = self.conv(x).flatten(1)              # [B, num_kernels*H*W]
        # Run FC1 in its own weight dtype with autocast disabled. For the online
        # encoder (fp32 weight) this avoids autocast's transient bf16 copy of the
        # ~2.5 GB weight on every forward (a ~1.2 GB peak-VRAM saving) and keeps the
        # leaked weight gradient at full fp32 precision (the CSF is tuned to the
        # fp32 floor, so an fp32 matmul is also the more faithful one). For the bf16
        # target encoder the cast is a no-op and it simply runs bf16, no extra copy.
        with torch.autocast(device_type=conv_out.device.type, enabled=False):
            h = self.activation(self.fc1(conv_out.to(self.fc1.weight.dtype)))  # [B, fc_size]
        out = self.fc2(h).view(-1, self.C, self.H, self.W)
        # Residual bypass keeps the benign model's input ~unchanged (stealth):
        # the leak lives entirely in FC1's gradient, not the forward signal.
        return x + out if residual else out


@dataclass
class LokiConfig:
    image_shape: Tuple[int, int, int] = (3, 32, 32)
    num_clients: int = 5
    local_dataset_size: int = 64          # images per client to leak in a round
    fc_multiplier: int = 4                # FC units per image (Sec. 3.3: ~4x)
    csf: float = 50.0                     # convolutional scaling factor (Eq. 10)
    fedavg: bool = True                   # Hardtanh + Eq. 5 scaling when True
    bias_mean: float = 0.0               # dataset-agnostic bias init (Sec. 4.1)
    bias_std: float = 0.5
    device: str = "cpu"
    seed: int = 12345


class Loki:
    """
    The LOKI attack. One instance owns the malicious FC structure (weights and
    bias cutoffs shared across clients) plus per-client identity mapping sets,
    and exposes the paper's mechanisms as methods.

    Typical use (see `attack`): the server builds one malicious model per client
    (each with that client's identity mapping set), each client trains the model
    it was sent on its private data, the per-client updates are securely
    aggregated, and the server de-aggregates and reconstructs a target client.
    """

    def __init__(self, config: Optional[LokiConfig] = None, **kwargs):
        self.cfg = config or LokiConfig(**kwargs)
        c = self.cfg
        self.device = torch.device(c.device)
        C, H, W = c.image_shape
        self.C, self.H, self.W = C, H, W

        # One identity mapping set = C kernels (one per input channel), so the
        # whole 3-channel image is pushed forward (Sec. 3.4). Full model
        # inconsistency: a distinct set per client => num_kernels = num_clients*C.
        self.set_size = C
        self.num_sets = c.num_clients
        self.num_kernels = self.num_sets * self.set_size

        # FC layer size scales with the per-client dataset, NOT the client count
        # (split scaling, Sec. 3.2): ~`fc_multiplier` neurons per image.
        self.fc_size = max(c.fc_multiplier * c.local_dataset_size, 2)

        # Bias cutoffs (one scalar per neuron), learned/refined across rounds.
        self.bias_mean = c.bias_mean
        self.bias_std = c.bias_std
        self.cutoffs: Optional[torch.Tensor] = None     # sorted ascending [fc_size]

        # Per-client key value kv (Eq. 10): kv_new = CSF * kv. Defaults to CSF.
        self.kv = c.csf if c.fedavg else 1.0

        self._template: Optional[LokiModule] = None

    # ------------------------------------------------------------------ #
    # 3.4 Convolutional parameters — identity mapping sets
    # ------------------------------------------------------------------ #
    def _set_identity_mapping(self, module: LokiModule, active_cid: Optional[int]) -> None:
        """
        Write the identity mapping sets into `module.conv`.

        If `active_cid` is given (model inconsistency / de-aggregation, Sec. 3.5),
        only that client's set is non-zero — every other kernel outputs zero, so
        the client's weight gradient lands in a disjoint block of FC1's input.
        If `active_cid is None`, all sets are written (no model inconsistency,
        Sec. 4.4): valid too, since the CSF still separates the blocks.
        """
        kc, cc = module.kernel_center()
        with torch.no_grad():
            module.conv.weight.zero_()
            sets = range(self.num_sets) if active_cid is None else [active_cid]
            for s in sets:
                for c in range(self.set_size):
                    oc = s * self.set_size + c       # output channel
                    # Single key value at the kernel centre on input channel c:
                    # copies channel c -> output channel oc (a spatial identity).
                    module.conv.weight[oc, c, kc, cc] = self.kv

    # ------------------------------------------------------------------ #
    # 2.3 / 3.6 FC layer parameters — binning weights, biases, Eq. 5 scaling
    # ------------------------------------------------------------------ #
    def setup_fc_biases(
        self,
        method: str = "agnostic",
        brightness: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Initialize the FC bias cutoffs (Sec. 4.1).

          * "observed": fit cutoffs to the quantiles of an observed average-pixel-
            intensity sample `brightness` — this is the distribution the server
            converges to via incremental learning, and the strongest setting
            (cf. "true initialization" in Table 3).
          * "agnostic": no prior knowledge — a normal init with mean `bias_mean`
            / std `bias_std` (Sec. 4.1 dataset-agnostic init).
          * "random": uniform over the observed range (weakest, Table 3).

        Cutoffs are stored sorted ascending so consecutive neurons form
        consecutive brightness bins.
        """
        n = self.fc_size
        g = torch.Generator().manual_seed(self.cfg.seed)
        if method == "observed":
            if brightness is None:
                raise ValueError("method='observed' needs a brightness sample")
            b = torch.sort(brightness.flatten().float().cpu())[0]
            self.bias_mean = float(b.mean())
            self.bias_std = float(b.std())
            # Place cutoffs *between* consecutive sorted brightness values so each
            # image falls in its own bin (the converged "true" distribution, the
            # strongest init in Table 3). Bin (cutoff_i, cutoff_{i+1}] then holds
            # a single image, which Eq. 9 reconstructs cleanly.
            margin = 0.01 * (b[-1] - b[0] + 1e-6)
            mids = (b[:-1] + b[1:]) / 2
            core = torch.cat([b[:1] - margin, mids, b[-1:] + margin])  # len = N+1
            if n <= core.numel():
                cutoffs = core[:n]
            else:
                # More neurons than images: pad with extra cutoffs spread over the
                # range (harmless empty bins — the split-scaling headroom).
                pad = torch.linspace(float(b[0] - margin), float(b[-1] + margin),
                                     n - core.numel())
                cutoffs = torch.cat([core, pad])
        elif method == "random":
            if brightness is None:
                lo, hi = -1.0, 1.0
            else:
                b = brightness.flatten().float().cpu()
                lo, hi = float(b.min()), float(b.max())
            cutoffs = torch.rand(n, generator=g) * (hi - lo) + lo
        else:  # agnostic
            cutoffs = torch.randn(n, generator=g) * self.bias_std + self.bias_mean

        self.cutoffs = torch.sort(cutoffs)[0]
        return self.cutoffs

    def learn_bias_distribution(self, activated_brightness: Sequence[float]) -> None:
        """
        Incremental distribution learning across rounds (Sec. 4.1): after a round,
        the server observes the brightness of neurons that reconstructed cleanly
        and updates the mean/std used to initialize biases next round.
        """
        if len(activated_brightness) == 0:
            return
        arr = np.asarray(activated_brightness, dtype=np.float64)
        self.bias_mean = float(arr.mean())
        self.bias_std = float(arr.std() + 1e-6)

    def _apply_fc_parameters(self, module: LokiModule) -> None:
        """
        Write the binning weights/biases into FC1 and the equal-row matrix into
        FC2, including the FedAVG inter-bias scaling (Eq. 5) and CSF (Eq. 10).
        """
        if self.cutoffs is None:
            self.setup_fc_biases("agnostic")
        cutoffs = self.cutoffs.to(self.device)
        n = self.fc_size

        # FC1 weight measures average pixel intensity m. The conv pushes the
        # image to a block of `C*H*W` outputs scaled by kv, so a uniform weight
        # of 1/(kv*C*H*W) over the inputs makes a neuron compute exactly m
        # (inactive blocks are zero and contribute nothing). This is W*/CSF of
        # Eq. 10: folding kv here keeps the measured value of m independent of CSF.
        base_w = 1.0 / (self.kv * self.C * self.H * self.W)

        # Bias is the negative cutoff: pre-activation = m - cutoff_i, which is
        # > 0 iff the image is brighter than cutoff_i (ReLU/Hardtanh active band).
        weight = torch.full((n, module.conv_out_elems), base_w, device=self.device)
        bias = -cutoffs.clone()

        if self.cfg.fedavg:
            # Eq. 5: scale weights and biases by the distance to the next bin so
            # the Hardtanh's fixed (0,1) non-zero-gradient band aligns with the
            # interval between adjacent cutoffs (consecutive bins).
            diffs = torch.diff(cutoffs)
            diffs = torch.cat([diffs, diffs[-1:]]).clamp_min(1e-6)   # [n]
            weight = weight / diffs.unsqueeze(1)
            bias = bias / diffs

        with torch.no_grad():
            module.fc1.weight.copy_(weight)
            module.fc1.bias.copy_(bias)
            # FC2: identical rows (Sec. 2.3) so dL/dW_i and dL/db_i share the same
            # downstream scale for every neuron. Small magnitude keeps the
            # residual bypass close to the original image (stealth).
            module.fc2.weight.fill_(1.0 / n)

    # ------------------------------------------------------------------ #
    # Build the malicious model for a given client
    # ------------------------------------------------------------------ #
    def build_module(self, active_cid: Optional[int]) -> LokiModule:
        """Construct the attack module to send to `active_cid` (None = all sets)."""
        module = LokiModule(self.cfg.image_shape, self.fc_size, self.num_kernels,
                            fedavg=self.cfg.fedavg).to(self.device)
        self._set_identity_mapping(module, active_cid)
        self._apply_fc_parameters(module)
        return module

    # ------------------------------------------------------------------ #
    # Client side (the only data-touching method): produce the update
    # ------------------------------------------------------------------ #
    def simulate_client_update(
        self,
        images: torch.Tensor,
        active_cid: int,
        downstream: Optional[nn.Module] = None,
        mode: str = "fedsgd",
        local_iters: int = 8,
    ) -> torch.Tensor:
        """
        Stand-in for a client training the malicious model it was sent on its
        private `images`. Returns FC1's weight *update* — the quantity the
        server receives (a gradient for FedSGD, Theta_t - Theta_{t+1} for
        FedAVG, Eq. 6). This is the boundary between client and server: only this
        method sees raw data.

        The leak is loss-agnostic (Sec. 3.5): the FC1 weight gradient is
        (dL/dy_i) * input for any scalar loss L, so a simple surrogate suffices
        when no `downstream` benign model is supplied.
        """
        images = images.to(self.device)
        module = self.build_module(active_cid)

        def loss_of(out_imgs: torch.Tensor) -> torch.Tensor:
            if downstream is not None:
                z = downstream(out_imgs)
                return z.float().pow(2).mean()
            # Surrogate: sum of the module output. Gives a constant dL/dy across
            # the output, so each activated neuron's gradient is exactly its
            # (scaled) input — the cleanest demonstration of the leak.
            return out_imgs.float().sum()

        if mode == "fedsgd":
            # One local iteration over the whole batch -> a single gradient (Eq. 1).
            module.zero_grad(set_to_none=True)
            loss = loss_of(module(images, residual=downstream is not None))
            loss.backward()
            return module.fc1.weight.grad.detach().clone()

        # FedAVG (Eq. 6): the server only sees Theta_t - Theta_{t+1}, the sum of
        # the per-local-iteration gradients. The convolutional scaling factor
        # (folded into kv / the FC weights, Eq. 10) is engineered so the leak
        # layer's weights stay ~fixed across local iterations (Eq. 7) and so an
        # image that already activated a neuron is pushed out of its Hardtanh band
        # and cannot re-activate it in a later iteration (Sec. 3.6). Under that
        # regime the parameter shift equals the accumulated per-mini-batch
        # gradient, which we form directly here — each image is seen once.
        module.zero_grad(set_to_none=True)
        bs = max(1, len(images) // max(1, local_iters))
        for it in range(local_iters):
            batch = images[it * bs:(it + 1) * bs]
            if len(batch) == 0:
                break
            loss_of(module(batch, residual=downstream is not None)).backward()
        return module.fc1.weight.grad.detach().clone()

    # ------------------------------------------------------------------ #
    # 3.5 De-aggregation & client identification
    # ------------------------------------------------------------------ #
    def _client_block(self, cid: int) -> slice:
        """Columns of FC1's weight that carry client `cid` (its conv channels)."""
        elems = self.set_size * self.H * self.W
        start = cid * elems
        return slice(start, start + elems)

    def identify_client(self, cid: int) -> int:
        """
        Sec. 4.2: the identity mapping set used to reconstruct a sample *is* the
        client id, so ownership survives aggregation/shuffling. Here it is the
        block index; exposed as a method to mirror the paper's framing.
        """
        return cid

    # ------------------------------------------------------------------ #
    # 3.5 Reconstruction (server side, no data access) — Eq. 8 / Eq. 9
    # ------------------------------------------------------------------ #
    def reconstruct(self, weight_grad: torch.Tensor, target_cid: int) -> torch.Tensor:
        """
        Reconstruct `target_cid`'s images from the (aggregated) FC1 weight update,
        restricted to that client's de-aggregated block (Eq. 8) and bias-free
        (Eq. 9 — scale to a maximum of 1, no bias gradient needed). Returns image
        fragments [n, C, H, W], one per bin.

        The per-neuron gradient differs by mode:
          * FedSGD (one-sided ReLU): neuron i accumulates every image brighter
            than cutoff_i, so the *consecutive-bin difference* W_i - W_{i+1}
            cancels the shared brighter images and isolates one input.
          * FedAVG (double-bounded Hardtanh + Eq. 5): neuron i's gradient is
            non-zero only for images in its own bin, so the *neuron weight
            gradient itself* already reconstructs the bin's image directly.
        """
        g = weight_grad.detach().to(self.device)            # [fc_size, in_features]
        block = g[:, self._client_block(target_cid)]        # [fc_size, C*H*W]
        elems = self.set_size * self.H * self.W
        rows = block[:, :elems] if self.cfg.fedavg else (block[:-1, :elems] - block[1:, :elems])

        # Sign-preserving Eq. 9 normalization, vectorized over all bins at once.
        # Scale each row by its own max-abs; empty rows are all-zero so dividing by
        # the clamped max leaves them zero, matching the old `recon if m>0 else vec`.
        m = rows.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)   # [n, 1]
        frags = (rows / m).view(-1, self.set_size, self.H, self.W)
        return frags

    def reconstruct_with_bias_count(
        self, weight_grad: torch.Tensor, bias_grad: torch.Tensor, target_cid: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Variant that also returns, per bin, the estimated number of contributing
        images from the bias gradient (Sec. 3.5). A bin with an estimated count of
        ~1 is a clean (single-image) reconstruction; >1 is a failed blend. The
        server uses this to filter without any ground truth.

        The per-bin "image mass" must be read the same way `reconstruct` defines a
        bin, which differs by mode:

          * FedAVG (Hardtanh): neuron i fires *only* for images in its own bin, so
            its bias gradient db_i is already the per-bin mass. Use it directly —
            one count per neuron, aligned with the fc_size fragments `reconstruct`
            returns. (Taking a consecutive difference here would instead measure
            count_i - count_{i+1}, which collapses two adjacent single-image bins
            to ~0 and mislabels them; see Sec. 3.6.)
          * FedSGD (ReLU): neuron i accumulates every image brighter than cutoff_i,
            so the per-bin mass is the consecutive difference db_i - db_{i+1},
            aligned with the fc_size-1 difference fragments `reconstruct` returns.
        """
        frags = self.reconstruct(weight_grad, target_cid)
        bg = bias_grad.detach().to(self.device)
        mass = bg.abs() if self.cfg.fedavg else (bg[:-1] - bg[1:]).abs()
        # One image's contribution = the smallest non-negligible bin (measured
        # relative to the busiest bin, so FP32 noise and empty bins are dropped).
        unit = mass[mass > mass.max() * 1e-3]
        unit = unit.min() if unit.numel() else torch.tensor(1.0, device=self.device)
        counts = (mass / unit.clamp_min(1e-12)).round()
        return frags, counts

    def reconstruct_with_fired(
        self, weight_grad: torch.Tensor, target_cid: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Like `reconstruct`, but also return a per-bin "fired" mass read from the
        de-aggregated WEIGHT block: the max-abs of each bin's Eq. 9 source row
        *before* the max-to-1 normalization. A bin whose mass is above the
        FP/noise floor captured an image for `target_cid`; empty bins are ~0.

        This is the multi-client-safe replacement for the bias-gradient count.
        Under secure aggregation the FC1 bias gradients are *summed across clients*
        (Sec. 3.5), so a per-client image signal is not recoverable from the bias.
        The weight block, however, IS de-aggregated per client via the identity
        mapping sets, so its per-bin energy is a correct per-client fired/empty
        signal — exactly what is needed to filter saved fragments and to pick the
        fired cutoffs for bias-distribution learning when extracting all clients.
        """
        g = weight_grad.detach().to(self.device)
        block = g[:, self._client_block(target_cid)]
        elems = self.set_size * self.H * self.W
        rows = block[:, :elems] if self.cfg.fedavg else (block[:-1, :elems] - block[1:, :elems])
        mass = rows.abs().amax(dim=1)                                  # [n]
        frags = (rows / mass.clamp_min(1e-12).unsqueeze(1)).view(-1, self.set_size, self.H, self.W)
        return frags, mass

    def bin_mass(self, weight_grad: torch.Tensor, target_cid: int) -> torch.Tensor:
        """Per-bin Eq.9 normalizer (max-abs of each de-aggregated source row) WITHOUT
        materializing the fragments -- the denominator `reconstruct` divides by.

        Same rows as `reconstruct`/`reconstruct_with_fired`, but only the amax is
        kept, so no multi-GB fragment tensor is built (matters on the memory-tight
        server). Paired with the client's logged DP eff.abs.noise_std it fixes the
        sandbox LEAK_NOISE (reconstruction_test.dp_noise_to_leak_noise)."""
        g = weight_grad.detach().to(self.device)
        block = g[:, self._client_block(target_cid)]
        elems = self.set_size * self.H * self.W
        rows = block[:, :elems] if self.cfg.fedavg else (block[:-1, :elems] - block[1:, :elems])
        return rows.abs().amax(dim=1)                                  # [n]

    @staticmethod
    def fired_mask(mass: torch.Tensor, rel_floor: float = 1e-3) -> torch.Tensor:
        """
        Bins whose weight-block energy exceeds `rel_floor` * the busiest bin
        captured an image; the rest are empty (FP noise). Returns a bool mask
        aligned 1:1 with the fragments from `reconstruct_with_fired`.
        """
        if mass.numel() == 0:
            return mass.bool()
        return mass > mass.max() * rel_floor

    # ------------------------------------------------------------------ #
    # Orchestration: full attack across clients, de-aggregate the target
    # ------------------------------------------------------------------ #
    def attack(
        self,
        client_images: Dict[int, torch.Tensor],
        target_cid: int = 0,
        downstream: Optional[nn.Module] = None,
        mode: str = "fedsgd",
        local_iters: int = 8,
        bias_init: str = "observed",
    ) -> torch.Tensor:
        """
        Run the end-to-end attack and return `target_cid`'s reconstructed
        fragments.

        Steps mirror a real round: (1) the server sets bias cutoffs (optionally
        from observed brightness of the target — the converged learned
        distribution); (2) each participating client trains the malicious model
        it was sent on its own data; (3) the server securely aggregates the
        updates (a plain sum here); (4) the server de-aggregates and reconstructs
        the target client (Eq. 8/9).
        """
        # (1) Bias setup. "observed" uses the target client's brightness, the
        # distribution the incremental learner converges to (Sec. 4.1).
        if bias_init == "observed":
            tgt = client_images[target_cid].to(self.device)
            brightness = tgt.mean(dim=(1, 2, 3))                  # avg pixel intensity
            self.setup_fc_biases("observed", brightness=brightness)
        else:
            self.setup_fc_biases(bias_init)

        # (2)+(3) Each client produces an update; secure aggregation = sum.
        agg = None
        for cid, imgs in client_images.items():
            upd = self.simulate_client_update(
                imgs, active_cid=cid, downstream=downstream,
                mode=mode, local_iters=local_iters,
            )
            agg = upd if agg is None else agg + upd

        # (4) De-aggregate + reconstruct the target (Eq. 8/9).
        return self.reconstruct(agg, target_cid)

    # ------------------------------------------------------------------ #
    # 4.3 Parameter / sparsity accounting
    # ------------------------------------------------------------------ #
    def parameter_overhead(self) -> Dict[str, int]:
        """
        Report the parameters the attack adds and how sparse they are (Sec. 4.3):
        only the C non-zero conv taps per client's set are needed, so the vast
        majority of the added conv parameters are zero.
        """
        conv = self.num_kernels * self.C * 3 * 3
        fc1 = self.fc_size * (self.num_kernels * self.H * self.W)
        fc2 = self.fc_size * (self.C * self.H * self.W)
        nonzero_conv = self.num_sets * self.set_size       # one tap per kernel used
        total = conv + fc1 + fc2
        return {
            "conv_params": conv,
            "fc1_params": fc1,
            "fc2_params": fc2,
            "total_added": total,
            "nonzero_conv_params": nonzero_conv,
            "conv_sparsity_pct": int(100 * (1 - nonzero_conv / max(1, conv))),
        }

    # ------------------------------------------------------------------ #
    # Evaluation helpers (use ground truth — for measuring the demo only)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_unit(img: torch.Tensor) -> torch.Tensor:
        """Min-max normalize a [C,H,W] image to [0,1] for comparison/display."""
        lo, hi = img.min(), img.max()
        return (img - lo) / (hi - lo + 1e-8)

    @classmethod
    def ssim(cls, a: torch.Tensor, b: torch.Tensor) -> float:
        """Global single-window SSIM on grayscale images in [0,1]."""
        a = cls._to_unit(a).mean(0)
        b = cls._to_unit(b).mean(0)
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        mu_a, mu_b = a.mean(), b.mean()
        va, vb = a.var(unbiased=False), b.var(unbiased=False)
        cov = ((a - mu_a) * (b - mu_b)).mean()
        num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
        den = (mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)
        return float(num / den)

    def evaluate_leakage(
        self, fragments: torch.Tensor, ground_truth: torch.Tensor, ssim_thresh: float = 0.5
    ) -> Dict[str, object]:
        """
        Greedily match each reconstructed fragment to its best ground-truth image
        by SSIM and count an image as *leaked* if SSIM > `ssim_thresh` (the
        paper's leakage criterion, Sec. 5.1). Returns the leakage rate and the
        per-ground-truth best fragment index (for display).
        """
        F_ = fragments.to(self.device)
        G = ground_truth.to(self.device)
        n_g = len(G)
        n_f = len(F_)

        # Vectorized global single-window SSIM matrix S[gi, fi]. The global SSIM is
        # just per-image mean/var plus the cross-covariance, so the whole n_g x n_f
        # grid is one matmul instead of n_g*n_f Python calls (orders of magnitude
        # faster, and runs on the GPU).
        def to_unit_flat(x: torch.Tensor) -> torch.Tensor:
            # Match Loki.ssim exactly: per-image min-max over *all* channels first
            # (_to_unit), then collapse to grayscale (channel mean), then flatten.
            n, c, h, w = x.shape
            flat_full = x.reshape(n, -1)                        # [N, C*H*W]
            lo = flat_full.min(dim=1, keepdim=True).values
            hi = flat_full.max(dim=1, keepdim=True).values
            unit = (flat_full - lo) / (hi - lo + 1e-8)
            gray = unit.reshape(n, c, h, w).mean(dim=1)         # [N, H, W]
            return gray.reshape(n, -1)                          # [N, H*W]

        Gf = to_unit_flat(G)                                    # [n_g, P]
        Ff = to_unit_flat(F_)                                   # [n_f, P]
        P = Gf.shape[1]
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        mu_g = Gf.mean(dim=1)                                   # [n_g]
        mu_f = Ff.mean(dim=1)                                   # [n_f]
        var_g = Gf.var(dim=1, unbiased=False)                  # [n_g]
        var_f = Ff.var(dim=1, unbiased=False)                  # [n_f]
        # cov[g,f] = E[gf] - mu_g*mu_f, with E[gf] = (Gf @ Ff^T) / P
        cov = (Gf @ Ff.t()) / P - mu_g[:, None] * mu_f[None, :]
        num = (2 * mu_g[:, None] * mu_f[None, :] + c1) * (2 * cov + c2)
        den = (mu_g[:, None] ** 2 + mu_f[None, :] ** 2 + c1) * \
              (var_g[:, None] + var_f[None, :] + c2)
        S = num / den                                          # [n_g, n_f]

        # Greedy one-to-one assignment by descending SSIM (same semantics as the
        # original), but driven off the matrix: repeatedly take the global argmax
        # and mask out its row/column. min(n_g, n_f) cheap iterations instead of
        # iterating a sorted list of n_g*n_f pairs.
        best_idx = [-1] * n_g
        best_ssim = [0.0] * n_g
        M = S.clone()
        neg = torch.finfo(M.dtype).min
        for _ in range(min(n_g, n_f)):
            flat = torch.argmax(M)
            gi = int(flat // n_f)
            fi = int(flat % n_f)
            val = float(M[gi, fi])
            if val <= neg / 2:
                break                                          # nothing left to assign
            best_idx[gi] = fi
            best_ssim[gi] = val
            M[gi, :] = neg
            M[:, fi] = neg
        leaked = sum(1 for s in best_ssim if s > ssim_thresh)
        return {
            "leaked": leaked,
            "total": n_g,
            "leakage_rate": leaked / max(1, n_g),
            "best_idx": best_idx,
            "best_ssim": best_ssim,
        }

    def save_round_fragments(
        self,
        fragments: torch.Tensor,
        root: str,
        server_round: int,
        counts: Optional[torch.Tensor] = None,
        clean_only: bool = True,
    ) -> None:
        """
        Save a whole round's fragments as a single file root/round_XXX.pt.

        The offline clustering (regroup_fragments) re-identifies each original
        across the mixed bins from scratch — the FC1 bin index is not a stable
        per-sample key — so the per-bin directory layout buys nothing downstream.
        Collapsing each round to one file replaces thousands of tiny per-bin .pt
        writes (and the matching reads) with a single sequential file, cutting
        fragment I/O and inode pressure several-fold without losing anything the
        clusterer consumes.

        Stores a dict:
            {"frags":  [n, C, H, W] float32,   # the kept fragments
             "bins":   [n] long,               # originating FC1 neuron (traceability)
             "counts": [n] long}               # bias_count of each kept bin
        With clean_only and `counts` given, every *non-empty* bin (count != 0) is
        kept and only empty bins (count 0) are dropped. The bias-mass count is an
        unreliable single-vs-blend signal under a real (non-uniform) downstream
        loss — it inflates the count of clean bins and silently discards good
        reconstructions — so blend rejection is deferred to the downstream quality
        filter. The per-bin `counts` are still recorded so a later stage can use
        them. Fragments stay float32 (no precision loss). Empty rounds write
        nothing.
        """
        if counts is not None and clean_only:
            keep = counts.detach().cpu() != 0
        else:
            keep = torch.ones(fragments.shape[0], dtype=torch.bool)
        sel = keep.nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            return
        os.makedirs(root, exist_ok=True)
        # Select the kept rows on-device, then move only those to CPU (avoids
        # copying the full [fc_size, C, H, W] fragment tensor off the GPU).
        kept = fragments.index_select(0, sel.to(fragments.device)).detach().cpu().float()
        payload = {"frags": kept, "bins": sel.cpu()}
        if counts is not None:
            payload["counts"] = counts.detach().cpu()[sel].long()
        # Per-fragment bias cutoff. A fragment is dL/dh_i * x_view / max|.| and
        # dL/dh_i has no fixed sign under a BYOL-style loss, so ~half are saved as
        # photographic negatives; the downstream min-max maps -x to 1-minmax(x),
        # a valid-looking image nothing flags. The cutoff is the sign reference
        # that undoes it: bin i fires only for views whose mean brightness lands
        # in its band, so sign(cutoff_i) is the sign the fragment mean must have.
        # This is the server's OWN chosen bias value, not a bias *gradient* (which
        # secure aggregation destroys, cf. reconstruct_with_fired) — so recording
        # it assumes nothing the extraction does not already assume.
        cutoffs = self.cutoffs
        if cutoffs is not None:
            if not self.cfg.fedavg:
                cutoffs = cutoffs[:-1]      # align to the difference fragments
            if cutoffs.numel() == fragments.shape[0]:
                payload["cutoffs"] = cutoffs.detach().cpu()[sel].float()
        torch.save(payload, os.path.join(root, f"round_{int(server_round):03d}.pt"))

    def save_fragments_by_bin(
        self,
        fragments: torch.Tensor,
        root: str,
        server_round: int,
        counts: Optional[torch.Tensor] = None,
        clean_only: bool = True,
    ) -> None:
        """
        Save each bin's fragment for this round under root/bin_XXXX/round_YYY.pt.

        A bin (FC1 neuron) captures images in a fixed brightness band, a stable
        property of the *underlying* image rather than its augmented view. So a
        bin's folder accumulates the multiple augmented views of (probably) the
        same original image across rounds -- exactly the per-image stack needed to
        invert the known augmentation pipeline one image at a time.

        If `counts` is given (the per-bin estimated number of contributing images
        from the bias gradient, Sec. 3.5), each bin folder also gets an appended
        info.csv row `round,bias_count`. A count of ~1 marks a clean single-image
        capture; >1 is a blend, so the offline reconstructor can filter/weight by
        it. `counts` is aligned 1:1 with the fragments (FedAVG: one per neuron;
        FedSGD: one per consecutive-bin difference), so every saved bin has a
        count; any trailing bin without one is recorded as empty.

        Storage control: when `counts` is given, `clean_only` keeps only the
        clean (count == 1) bins on disk -- the blended/empty bins are exactly
        what the offline reconstructor discards anyway, and they are ~70% of
        bins, so skipping them cuts the fragment dump several-fold without losing
        anything it consumes. Bin indices stay stable (a bin is the same FC1
        neuron every round), so a bin only accumulates round_*.pt for the rounds
        in which it was clean. With `counts=None` (no diagnostic), all bins are
        written as before.
        """
        n_counts = 0 if counts is None else int(counts.numel())
        for i in range(fragments.shape[0]):
            c = int(counts[i].item()) if (counts is not None and i < n_counts) else None
            # Storage filter: persist every non-empty bin (count != 0), dropping
            # only empty bins. The bias-mass count is an unreliable single-vs-blend
            # signal under a real downstream loss (it inflates clean-bin counts), so
            # blend rejection is deferred to the downstream quality filter.
            if clean_only and counts is not None and (c is None or c == 0):
                continue
            bdir = os.path.join(root, f"bin_{i:04d}")
            os.makedirs(bdir, exist_ok=True)
            torch.save(
                fragments[i].detach().cpu(),
                os.path.join(bdir, f"round_{int(server_round):03d}.pt"),
            )
            if counts is not None:
                info = os.path.join(bdir, "info.csv")
                new_file = not os.path.exists(info)
                with open(info, "a") as f:
                    if new_file:
                        f.write("round,bias_count\n")
                    f.write(f"{int(server_round)},{'' if c is None else c}\n")

    def save_fragments(
        self,
        fragments: torch.Tensor,
        path: str,
        ground_truth: Optional[torch.Tensor] = None,
        eval_result: Optional[Dict[str, object]] = None,
        max_show: int = 64,
    ) -> None:
        """
        Save a PNG grid of reconstructed fragments (and, if provided, the matched
        ground-truth images above them). De-normalizes via CIFAR stats only for
        viewing. Requires matplotlib (already a project dependency).
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if ground_truth is not None and eval_result is not None:
            idx = eval_result["best_idx"]
            gts = ground_truth[: max_show]
            recons = [fragments[i] if i >= 0 else torch.zeros_like(fragments[0])
                      for i in idx[: max_show]]
            recons = torch.stack(recons)
            n = len(gts)
            cols = min(8, n)
            rows = math.ceil(n / cols)
            fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 1.3, rows * 2.6))
            axes = np.atleast_2d(axes)
            for r in range(rows):
                for c in range(cols):
                    k = r * cols + c
                    ax_g = axes[r * 2][c]
                    ax_r = axes[r * 2 + 1][c]
                    ax_g.axis("off"); ax_r.axis("off")
                    if k < n:
                        ax_g.imshow(self._to_unit(gts[k]).permute(1, 2, 0).cpu().numpy())
                        ax_r.imshow(self._to_unit(recons[k]).permute(1, 2, 0).cpu().numpy())
            fig.suptitle(
                f"LOKI reconstruction — leaked {eval_result['leaked']}/{eval_result['total']} "
                f"({eval_result['leakage_rate']*100:.1f}%)  [top: ground truth, bottom: reconstruction]"
            )
        else:
            show = fragments[: max_show]
            n = len(show)
            cols = min(8, n)
            rows = math.ceil(n / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.3, rows * 1.3))
            axes = np.atleast_2d(axes)
            for r in range(rows):
                for c in range(cols):
                    k = r * cols + c
                    ax = axes[r][c]; ax.axis("off")
                    if k < n:
                        ax.imshow(self._to_unit(show[k]).permute(1, 2, 0).cpu().numpy())
            fig.suptitle("LOKI reconstructed fragments")
        fig.tight_layout()
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)

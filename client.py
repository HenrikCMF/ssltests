import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import torch
import flwr as fl
from pathlib import Path
from dataloader import CIFAR10BYOLClientData, CIFAR100BYOLClientData, TinyImageNetBYOLClientData
from architectures import build_models
from SSL import SimpleBYOL
from dp_accounting import noise_std_for_eps
import pickle

DATA_DIR = Path(__file__).resolve().parent / "data"

# Reference batch size at which the base learning rate was tuned. The LR is
# scaled linearly off this (lr = base_lr * batch_size / REF_BATCH), so larger
# batches stay well-tuned. BYOL's base LR was tuned at 128.
BYOL_REF_BATCH = 128

# CIFAR-10 download/integrity check is only needed once per worker process; after
# the first client is built we pass download=False to skip torchvision's repeated
# ~390ms MD5 verification on every subsequent round.
_DATASET_VERIFIED = False

# Per-process cache of client data objects (loaders + workers + partition) keyed
# by the config that determines them. Flower rebuilds FedClient every round, but
# the data object is a deterministic function of cid + config, so we build it once
# per process and reuse it — avoiding the ~579ms CIFAR reload and worker respawn.
_DATA_CACHE = {}

# Per-process cache of the built (and, when use_compile, torch.compiled) model
# triple (online, target, predictor), keyed on the architecture signature. Flower
# reuses each Ray actor process across rounds, so building + compiling ONCE here
# and handing the SAME module objects to every round's trainer is what lets
# torch.compile keep its graph: the weight-update paths (SimpleBYOL._set_sd_floats
# and load_state_dict) copy in place, preserving every parameter tensor's identity
# and metadata, so no Dynamo guard is invalidated and no recompile happens after
# round 1. Capacity 1 (the arch key is constant within a run): on any signature
# change the old triple is dropped first, so at most one — possibly large —
# compiled model set is resident per actor. That also removes the model-stacking
# release_gpu guarded against, since each client loads its weights into these same
# resident tensors instead of allocating a fresh model.
_MODEL_CACHE = {}


def _build_or_get_models(dataset: str, embedding_size: int, loki: bool,
                         loki_fc_size: int, loki_num_kernels: int,
                         loki_image_shape, use_compile: bool):
    """Return the (online, target, predictor) triple for this architecture.

    With use_compile the triple is built + compiled once per process and reused
    every round (see _MODEL_CACHE). Without it, a fresh triple is built each round
    and nothing is cached — byte-for-byte the pre-existing behavior."""
    if not use_compile:
        return build_models(
            emb_dim=embedding_size, loki=loki, loki_image_shape=loki_image_shape,
            loki_fc_size=loki_fc_size, loki_num_kernels=loki_num_kernels,
        )

    key = (dataset, embedding_size, loki, loki_fc_size, loki_num_kernels)
    triple = _MODEL_CACHE.get(key)
    if triple is None:
        # Drop any resident triple for a different arch first so we never hold two
        # (compiled) model sets on the GPU at once.
        _MODEL_CACHE.clear()
        model, ema, predictor = build_models(
            emb_dim=embedding_size, loki=loki, loki_image_shape=loki_image_shape,
            loki_fc_size=loki_fc_size, loki_num_kernels=loki_num_kernels,
        )
        # Compile IN PLACE (nn.Module.compile) rather than reassigning to a
        # torch.compile(...) wrapper: in place keeps the same object, class and
        # state_dict keys, so _sd_float_keys / get_parameters / the server's
        # array-count split / the DP key mapping all stay byte-for-byte identical.
        # dynamic=False locks the fixed (drop_last=True) batch shape to one static
        # graph. Compilation is lazy — it traces on the first CUDA forward inside
        # train(), after SimpleBYOL has moved these to the GPU. The online encoder
        # and predictor are always uniform fp32 and safe to compile; the target
        # encoder is compiled only when NOT loki, because a loki target carries a
        # deliberately bf16 trap sub-module (mixed dtype) and is left eager.
        model.compile(dynamic=False)
        predictor.compile(dynamic=False)
        if not loki:
            ema.compile(dynamic=False)
        triple = (model, ema, predictor)
        _MODEL_CACHE[key] = triple
    return triple


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()


class FedClient(fl.client.NumPyClient):
    def __init__(self, cid: int, num_partitions: int, local_epochs: int,
                 batch_size: int, total_rounds: int, embedding_size: int,
                 byol_base_lr: float = 0.032,
                 loki: bool = False, loki_fc_size: int = 1024,
                 loki_num_kernels: int = 3,
                 dataset: str = "cifar10", classes_per_client: int = 2,
                 data_fraction: float = 1.0,
                 strong_aug: bool = False,
                 dp: bool = False,
                 dp_noise_multiplier: float = 0.0,
                 dp_scheme: str = "per_layer",
                 dp_global_std: float = 0.0,
                 dp_eps: float = 0.0,
                 dp_delta: float = 1e-5,
                 dp_clip: float = 0.0,
                 dp_calibrate: bool = False,
                 use_autoscaler: bool = False,
                 use_compile: bool = False,
                 use_async_save: bool = False):
        self.cid          = int(cid)
        self.local_epochs = local_epochs
        # Rounds this client will participate in. fraction_fit=1.0 means every
        # client is sampled every round, so this is exactly the number of
        # compositions the "eps" scheme must account for (no amplification).
        self.total_rounds = int(total_rounds)
        self.tau          = 0.7
        self.lambda_k     = None
        self.NUM_ROUND    = 0.0
        # Differential-privacy mode. When True, every client perturbs the model
        # update it transmits (update = local - global) before sending it. The
        # client is blind to any server-injected structure (e.g. a LOKI trap), so
        # it applies ONE uniform rule to every parameter array: add Gaussian noise
        # whose L2 norm equals dp_noise_multiplier x that array's own update norm
        # (see _apply_dp). Scaling per layer is essential here — a single GLOBAL
        # clip or noise is set by LOKI's ~1e5-scaled trap layer, which drives the
        # real encoder/predictor update to zero; per-layer relative noise instead
        # perturbs every layer by the same fraction, so the honest model keeps
        # learning while the trap channel LOKI reads is degraded. dp_noise_multiplier
        # is the adjustable "amount of noise" knob (0 disables it); with relative
        # scaling it needs no clip-norm calibration. Mirrors the strong_aug defense
        # toggle but acts on the transmitted weights, not the input views.
        self.dp                  = bool(dp)
        self.dp_noise_multiplier = float(dp_noise_multiplier)
        # DP scheme selector (see _apply_dp):
        #   "eps"       — the only scheme with a real (eps, delta) guarantee. Clips
        #                 the WHOLE update vector to dp_clip, then adds the Gaussian
        #                 noise that dp_accounting calibrates for dp_eps over
        #                 total_rounds. Client-level DP in the local model (the
        #                 server is the adversary, so it cannot be trusted to add
        #                 the noise itself). Sweep dp_eps to trade extraction
        #                 success against accuracy.
        #   "per_layer" — LEGACY, NOT (eps, delta)-DP. noise L2 = dp_noise_multiplier
        #                 * ||u_layer||, scaled to EACH layer's own update norm. Has
        #                 no epsilon because (a) nothing is clipped, so sensitivity is
        #                 unbounded, and (b) the noise scale is itself a function of
        #                 the private data. Kept so earlier runs stay reproducible.
        #   "global"    — LEGACY, NOT (eps, delta)-DP. One absolute std on every
        #                 weight (N(0, dp_global_std^2)), no clip => unbounded
        #                 sensitivity => no epsilon. Kept for the earlier fixed-std
        #                 sweep; dp_accounting.eps_for_noise_std can place those runs
        #                 on the eps axis after the fact IF a clip norm is named.
        self.dp_scheme           = str(dp_scheme)
        self.dp_global_std       = float(dp_global_std)
        self.dp_eps              = float(dp_eps)
        self.dp_delta            = float(dp_delta)
        self.dp_clip             = float(dp_clip)
        # Calibration mode: measure and report the update norms, transmit the
        # update UNCHANGED (no clip, no noise). Exists because DP_CLIP cannot be
        # read off a live eps run — sigma scales with C, so widening C to disable
        # the clip also inflates the noise. The run is therefore a plain no-DP run
        # plus logging, which is exactly what a public a-priori choice of C should
        # be based on. NEVER report a run made in this mode as private.
        self.dp_calibrate        = bool(dp_calibrate)
        # Calibrate eps -> noise std ONCE. This must not depend on the data (a
        # data-dependent noise scale is exactly what voids the legacy schemes'
        # claim), so it is fixed here from public hyperparameters only.
        self.dp_sigma = 0.0
        if self.dp and self.dp_scheme == "eps" and not self.dp_calibrate:
            if self.dp_clip <= 0.0:
                raise ValueError(
                    'DP_SCHEME="eps" requires DP_CLIP > 0: with no clip the update '
                    "has unbounded L2 sensitivity and no epsilon exists."
                )
            self.dp_sigma = noise_std_for_eps(
                eps=self.dp_eps, delta=self.dp_delta,
                rounds=self.total_rounds, clip=self.dp_clip,
            )
        # Fresh RNG per client instance. FedClient is rebuilt every round, so this
        # draws new OS entropy each round — noise must never repeat across rounds.
        self._dp_rng = np.random.default_rng()
        # FedEMA divergence-aware autoscaler toggle. When False, blending uses
        # mu = 1 (the pure-global fast path == FedBYOL). When True, mu is the
        # dynamic per-client value from paper Eq. 3 and the corrected blend runs.
        self.use_autoscaler = bool(use_autoscaler)
        # torch.compile the encoders/predictor once per process and reuse them
        # across rounds so the compiled graph never has to be rebuilt. See
        # _build_or_get_models / _MODEL_CACHE and SimpleBYOL.release_gpu.
        self.use_compile = bool(use_compile)
        # Overlap the ~2.25 GB per-client model save with the next client's
        # load+train by writing it on a background thread (SimpleBYOL.save).
        self.use_async_save = bool(use_async_save)
        # When True the BYOL encoder carries a (gated) LOKI attack module at its
        # input; the malicious server arms it for this client only if it is the
        # attack target (see server.FedEMAStrategyWithKnn).
        self.loki             = bool(loki)
        self.loki_fc_size     = loki_fc_size
        self.loki_num_kernels = loki_num_kernels

        self._weight_prefix = f"local_weights/client{self.cid}"
        self._lambda_path   = f"local_weights/lambda{self.cid}.pkl"

        # Reuse the per-process data object across rounds (see _DATA_CACHE).
        cache_key = (self.cid, num_partitions, batch_size, dataset, data_fraction, strong_aug)
        data_obj = _DATA_CACHE.get(cache_key)
        if data_obj is None:
            global _DATASET_VERIFIED
            download = not _DATASET_VERIFIED
            _DATASET_VERIFIED = True

            _dataset_cls = {
                "tiny_imagenet": TinyImageNetBYOLClientData,
                "cifar100": CIFAR100BYOLClientData,
            }.get(dataset, CIFAR10BYOLClientData)
            data_obj = _dataset_cls(
                num_clients=num_partitions,
                cid=self.cid,
                classes_per_client=classes_per_client,
                batch_size=batch_size,
                num_workers=4,
                keep_labels=False,
                data_dir="./data",
                seed=12345,
                device=DEVICE,
                download=download,
                data_fraction=data_fraction,
                strong_aug=strong_aug,
            )
            _DATA_CACHE[cache_key] = data_obj
        self.train_loader, self.val_loader = data_obj.get_loaders()

        # Linear LR scaling rule (BYOL): lr grows with batch size off a
        # reference of BYOL_REF_BATCH, so larger batches stay well-tuned.
        byol_lr = byol_base_lr * batch_size / BYOL_REF_BATCH
        loki_image_shape = (3, 64, 64) if dataset == "tiny_imagenet" else (3, 32, 32)
        model, ema, predictor = _build_or_get_models(
            dataset=dataset,
            embedding_size=embedding_size,
            loki=self.loki,
            loki_fc_size=self.loki_fc_size,
            loki_num_kernels=self.loki_num_kernels,
            loki_image_shape=loki_image_shape,
            use_compile=self.use_compile,
        )
        self.trainer = SimpleBYOL(
            online_encoder=model,
            target_encoder=ema,
            predictor=predictor,
            lr=byol_lr,
            moving_average_decay=0.99,
            use_ema=True,
            local_epochs=local_epochs,
            dataset_len=len(self.train_loader),
            total_rounds=total_rounds,
            persist_models=self.use_compile,
            use_async_save=self.use_async_save,
        )

    # ------------------------------------------------------------------ #
    # Flower interface
    # ------------------------------------------------------------------ #

    def fit(self, parameters, config):
        selected_prev  = bool(int(config.get("selected_prev", 0)))
        self.NUM_ROUND = float(config.get("server_round", -1.0))

        has_local = self._load_local()

        # Lambda is now computed and distributed by the server (paper Alg. 1;
        # see server.FedEMAStrategy.aggregate_fit / configure_fit). The client
        # just consumes the value the server sent in this round's config; a
        # negative value is the server's "null" (uncalibrated) sentinel. This is
        # read after _load_local so the server value always takes precedence over
        # any stale local copy.
        lam_cfg = float(config.get("lambda_k", -1.0))
        self.lambda_k = None if lam_cfg < 0 else lam_cfg

        if not has_local or not selected_prev:
            # Case A: first round, or not selected last round — hard reset to global.
            self.trainer.reset_to_global(parameters)
        else:
            # Case B: fuse the local model with the freshly received global one.
            # Toggle on `mu`:
            #   * use_autoscaler OFF -> mu = 1: the existing pure-global fast path
            #     (FedBYOL), unchanged.
            #   * use_autoscaler ON  -> mu != 1: FedEMA (paper Alg. 1, Eq. 1-3).
            if self.use_autoscaler:
                # Measure divergence FIRST, without mutating the model, so mu can
                # be chosen before blending (the Eq.3 -> Eq.1 order).
                div = self.trainer.divergence_from_global(parameters)
                # Lambda calibration now happens on the server; local computation
                # disabled.
                # if self.lambda_k is None:
                #     # Autoscaler calibration, done once: makes mu_local == tau here.
                #     self.lambda_k = self.tau / (div + 1e-12)
                # Paper mu is the KEEP-LOCAL weight; blend_with_global's mu is the
                # KEEP-GLOBAL weight, so pass its complement. div>0 keeps mu<1, so
                # this never collides with the mu==1 fast path.
                mu_local = min(self.lambda_k * div, 1.0)
                mu = 1.0 - mu_local
            else:
                mu = 1  # FedBYOL: pure global

            div = self.trainer.blend_with_global(parameters, mu=mu)
            # Lambda is set by the server now; local computation disabled.
            # if self.lambda_k is None:
            #     # FedBYOL branch still records a lambda_k from the divergence.
            #     self.lambda_k = self.tau / div

            if self.cid == 0:
                print(f"r{int(self.NUM_ROUND)} div={div:.4f} mu(global)={mu:.4f} "
                      f"lambda={self.lambda_k}")

        train_loss = self.trainer.train(
            self.train_loader, epochs=self.local_epochs, server_round=self.NUM_ROUND
        )
        self._save_local()
        # DIAGNOSTIC: per-client worker RSS to localize the steady host-RAM climb.
        #from server import log_mem
        #log_mem(f"client {self.cid} fit r{int(self.NUM_ROUND)}")
        out_params = self.trainer.get_parameters()
        if self.dp:
            # Privatise the transmitted update (local - global) via the Gaussian
            # mechanism. `parameters` is the global model this round, aligned
            # array-for-array with get_parameters() (same encoder+predictor split).
            out_params = self._apply_dp(parameters, out_params)
        result = (
            out_params,
            len(self.train_loader.dataset),
            # Report the partition id so the (malicious) server can map Flower's
            # opaque ClientProxy ids to partitions and target a specific client.
            {"train_loss": train_loss, "cid": self.cid},
        )
        # The result above is CPU numpy and all state is on disk (_save_local), so
        # this trainer's GPU models/grad/optimizer are now disposable. The actor
        # runs the clients sequentially and rebuilds a fresh trainer next round;
        # releasing here keeps the next client from stacking its ~5 GB on top of
        # this one (the round-13 OOM was that doubled high-water mark).
        self.trainer.release_gpu()
        return result

    def evaluate(self, parameters, config):
        self.trainer.set_parameters(parameters)
        loss = 1
        return float(loss), len(self.val_loader.dataset), {"val_loss": float(loss)}

    # ------------------------------------------------------------------ #
    # Differential privacy
    # ------------------------------------------------------------------ #

    def _apply_dp(self, global_params, local_params):
        """Per-layer relative Gaussian perturbation of the transmitted update.

        For each parameter array the update u = local - global is perturbed by
        i.i.d. Gaussian noise whose L2 norm equals sigma * ||u|| (sigma =
        self.dp_noise_multiplier), then re-centred on the global model. Because
        the noise is scaled to each layer's own update norm, one sigma perturbs
        every layer by the same fraction of its update — the huge CSF-scaled LOKI
        trap and the small real weights alike — so the honest model keeps learning
        while the trap-delta channel LOKI reads is degraded. A single global clip
        or global noise cannot do this: LOKI's ~1e5-scaled trap dominates the norm
        and any usable noise level then swamps the real update.

        Note: this is an empirical relative-noise defense, not formally-accounted
        (eps, delta)-DP — a real privacy budget is unattainable at 5 clients anyway.
        Math runs in float64; each array is cast back to its float32 dtype.
        """
        if self.dp_scheme == "eps":
            return self._apply_dp_eps(global_params, local_params)

        sigma  = self.dp_noise_multiplier
        glob   = (self.dp_scheme == "global")
        g_std  = self.dp_global_std
        out, norms, sizes = [], [], []
        for l, g in zip(local_params, global_params):
            u = l.astype(np.float64) - g.astype(np.float64)
            n = float(np.sqrt(np.sum(u * u)))
            norms.append(n)
            sizes.append(u.size)
            if glob:
                # "global": ONE absolute per-coordinate std on every layer alike.
                if g_std > 0.0:
                    u = u + self._dp_rng.normal(0.0, g_std, size=u.shape)
            elif sigma > 0.0 and n > 0.0:
                # "per_layer": noise L2 norm == sigma * this layer's own norm.
                std = sigma * n / np.sqrt(u.size)
                u = u + self._dp_rng.normal(0.0, std, size=u.shape)
            out.append((g.astype(np.float64) + u).astype(g.dtype))
        if self.cid == 0 and norms:
            # The dominant-norm array is the LOKI trap fc1 in practice; its
            # effective ABSOLUTE per-coordinate noise std is what the LOKI paper's
            # sigma axis (Fig. 12) measures, so log it to compare on equal footing.
            j = int(np.argmax(norms))
            if glob:
                # global: every layer gets the SAME absolute std g_std.
                print(f"r{int(self.NUM_ROUND)} dp(GLOBAL): abs.std={g_std:g} on ALL "
                      f"layers  |update| min={min(norms):.4g} max={norms[j]:.4g}  "
                      f"dominant-layer d={sizes[j]:g} "
                      f"[one std for a ~7-decade layer spread -> see layer-RMS]")
            else:
                eff_std = sigma * norms[j] / np.sqrt(sizes[j]) if sizes[j] else 0.0
                print(f"r{int(self.NUM_ROUND)} dp(per-layer): sigma={sigma:g}  "
                      f"|update| min={min(norms):.4g} max={norms[j]:.4g}  "
                      f"dominant-layer d={sizes[j]:g}  "
                      f"eff.abs.noise_std(that layer)={eff_std:.4g} "
                      f"[vs LOKI paper's absolute sigma]")
            # Per-layer update RMS distribution -- the evidence for/against "just use
            # low global DP". Global DP is one absolute noise s for all layers; it can
            # both spare honest learning AND corrupt the leak only if the leak signal
            # (~m_fired median ~1e-3) sits BELOW every honest layer's RMS (a clean gap).
            # If honest layers reach down to/below ~1e-3, the scales overlap and no
            # single s works. Count layers per decade to see the spread directly.
            rms = np.array([n / np.sqrt(s) for n, s in zip(norms, sizes) if n > 0.0 and s > 0])
            if rms.size:
                pct = np.percentile(rms, [0, 10, 50, 90, 100])
                below = {t: int((rms < t).sum()) for t in (1e-4, 1e-3, 1e-2)}
                print(f"r{int(self.NUM_ROUND)} dp(layer-RMS n={rms.size}): "
                      f"min={pct[0]:.3g} p10={pct[1]:.3g} median={pct[2]:.3g} "
                      f"p90={pct[3]:.3g} max={pct[4]:.3g}  "
                      f"#layers<1e-4={below[1e-4]} <1e-3={below[1e-3]} <1e-2={below[1e-2]} "
                      f"[overlap w/ leak ~1e-3 => global DP can't gap it]")
        return out

    def _apply_dp_eps(self, global_params, local_params):
        """Client-level local DP with a real (eps, delta) guarantee.

        The mechanism, in order:
          1. u = local - global, over the FULL parameter vector (all arrays).
          2. Clip: u <- u * min(1, C/||u||_2). This is the step the legacy
             schemes lack, and it is what creates a bounded L2 sensitivity —
             without it no epsilon exists at any noise level.
          3. Add N(0, sigma^2) to every coordinate, sigma calibrated from
             dp_eps in __init__ (see dp_accounting).
          4. Re-centre on the global model and send.

        Steps 2-4 are post-processing of the client's data, so the released
        update is (dp_eps, dp_delta)-DP at CLIENT level against the server over
        all total_rounds rounds combined — not per round.

        Caveat worth knowing when reading the sweep: the clip is applied to the
        whole vector, and LOKI's CSF-scaled trap dominates ||u||. So a C small
        enough to be a meaningful bound also scales the honest weights down by
        the same factor, acting like a large LR cut. That coupling is a property
        of flat clipping under LOKI, not an accounting error.
        """
        updates = [l.astype(np.float64) - g.astype(np.float64)
                   for l, g in zip(local_params, global_params)]
        total_norm = float(np.sqrt(sum(float(np.sum(u * u)) for u in updates)))

        if self.dp_calibrate:
            if self.cid == 0:
                self._report_calibration(updates, total_norm)
            return local_params

        factor = min(1.0, self.dp_clip / total_norm) if total_norm > 0.0 else 1.0

        out = []
        for u, g in zip(updates, global_params):
            u = u * factor + self._dp_rng.normal(0.0, self.dp_sigma, size=u.shape)
            out.append((g.astype(np.float64) + u).astype(g.dtype))

        if self.cid == 0:
            print(f"r{int(self.NUM_ROUND)} dp(eps): eps={self.dp_eps:g} "
                  f"delta={self.dp_delta:g} rounds={self.total_rounds} "
                  f"C={self.dp_clip:g} sigma={self.dp_sigma:.4g}  "
                  f"|update|={total_norm:.4g} clip_factor={factor:.4g}")
            if factor > 0.999:
                print(f"r{int(self.NUM_ROUND)} dp(eps): note clip inactive "
                      f"(|update| < C) -> sensitivity bound is loose; a smaller C "
                      f"would buy the same eps with less noise")

            # float32 transmission fidelity. Parameters go over the wire as
            # ABSOLUTE weights (g + u), not as u, so each layer's update is
            # rounded at the quantum of |g| — and on the CSF-scaled trap layer
            # |g| ~ 1e5 puts that quantum near 1e-2. Clipping multiplies every
            # update by `factor` (~1e-4 once the trap dominates ||u||), which can
            # push both the honest deltas and the noise under that quantum.
            #
            # This does NOT void the epsilon: rounding is a data-independent
            # function of an already-noised value, i.e. post-processing, so the
            # guarantee survives it. What it does void is the EXPERIMENT — if the
            # transmitted update is mostly quantization error, the run measures
            # float32 rounding rather than the DP mechanism, and any drop in
            # extraction success cannot be attributed to eps.
            realized = float(np.sqrt(sum(
                float(np.sum((o.astype(np.float64) - g.astype(np.float64)) ** 2))
                for o, g in zip(out, global_params))))
            d = sum(u.size for u in updates)
            expected = float(np.sqrt((factor * total_norm) ** 2 + d * self.dp_sigma ** 2))
            if expected > 0.0 and realized < 0.99 * expected:
                lost = 100.0 * (1.0 - realized / expected)
                print(f"r{int(self.NUM_ROUND)} dp(eps): *** WARNING float32 "
                      f"transmission lost {lost:.1f}% of the update norm "
                      f"(realized={realized:.4g} vs expected={expected:.4g}) -> "
                      f"clipped update/noise is below the float32 quantum on the "
                      f"large-|g| layers. eps still holds (post-processing), but "
                      f"utility/extraction at this eps are CONFOUNDED by rounding; "
                      f"raise C or lower LOKI_CSF before reading the sweep ***")
        return out

    def _report_calibration(self, updates, total_norm):
        """Report what a flat clip would be bounding, and what it would cost.

        The decisive number is `honest |update|` vs the total: flat clipping scales
        the WHOLE vector by C/||u||, so whatever fraction of the norm the LOKI trap
        owns is the factor by which the honest weights get crushed for free. If the
        trap owns essentially all of it, no C both bounds sensitivity and leaves the
        real model trainable, and that is a property of the attack, not of the
        accounting.
        """
        # Name the arrays: get_parameters() is encoder state-dict floats followed by
        # predictor state-dict floats, so the same key order reproduces the layout.
        try:
            keys = (self.trainer._sd_float_keys(self.trainer.online_encoder)
                    + self.trainer._sd_float_keys(self.trainer.predictor))
        except Exception:
            keys = [f"array[{i}]" for i in range(len(updates))]
        if len(keys) != len(updates):
            keys = [f"array[{i}]" for i in range(len(updates))]

        norms = [float(np.sqrt(np.sum(u * u))) for u in updates]
        # The trap tensors live under the encoder's LOKI module; everything else is
        # the honest model whose training we care about preserving.
        is_trap = [("loki" in k.lower() or "trap" in k.lower()) for k in keys]
        honest_sq = sum(n * n for n, t in zip(norms, is_trap) if not t)
        trap_sq = sum(n * n for n, t in zip(norms, is_trap) if t)
        honest = float(np.sqrt(honest_sq))
        trap = float(np.sqrt(trap_sq))

        print(f"\n[DP-CAL] r{int(self.NUM_ROUND)} total |update| = {total_norm:.6g}"
              f"   (NOT private: no clip, no noise applied)")
        if any(is_trap):
            share = 100.0 * trap_sq / (total_norm ** 2) if total_norm > 0 else 0.0
            print(f"[DP-CAL] r{int(self.NUM_ROUND)} LOKI-trap |update| = {trap:.6g} "
                  f"({share:.4f}% of the squared norm)   "
                  f"honest |update| = {honest:.6g}")
            print(f"[DP-CAL] r{int(self.NUM_ROUND)} => C = {total_norm:.4g} leaves the "
                  f"clip inactive; C = {honest:.4g} (honest-only scale) would scale the "
                  f"honest update by {min(1.0, honest / total_norm):.4g}")
        order = sorted(range(len(norms)), key=lambda i: norms[i], reverse=True)[:6]
        print(f"[DP-CAL] r{int(self.NUM_ROUND)} top arrays by |update|:")
        for i in order:
            print(f"[DP-CAL]     {norms[i]:>12.6g}  {str(updates[i].shape):>18}  {keys[i]}")

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _load_local(self) -> bool:
        ok = self.trainer.load(self._weight_prefix)
        # Lambda is now computed and distributed by the server, so it is no longer
        # loaded from local disk here; fit() overwrites self.lambda_k from the
        # server-sent config right after this call. Local load disabled.
        # if ok:
        #     if self.NUM_ROUND > 2:
        #         try:
        #             with open(self._lambda_path, "rb") as f:
        #                 self.lambda_k = pickle.load(f)
        #         except Exception:
        #             self.lambda_k = None
        #     else:
        #         self.lambda_k = None
        # else:
        #     self.lambda_k = None
        return ok

    def _save_local(self) -> None:
        self.trainer.save(self._weight_prefix)
        # Lambda is owned by the server now; local persistence disabled.
        # if self.lambda_k is not None:
        #     with open(self._lambda_path, "wb") as f:
        #         pickle.dump(self.lambda_k, f)


# Kept for backward compatibility with FedEMA_run.py (used by sd_float_arrays_bn).
def _state_dict_keys_float(module: torch.nn.Module):
    sd = module.state_dict()
    return [k for k, v in sd.items()
            if torch.is_tensor(v) and v.is_floating_point()
            and "running_mean" not in k and "running_var" not in k]


if __name__ == "__main__":
    pass

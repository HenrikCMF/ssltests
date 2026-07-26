import io
import gc
import threading
import concurrent.futures
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

# ---- Background model-save infrastructure (client-side I/O overlap) ---------- #
# torch.save of a client's ~2.25 GB state (the LOKI trap fc1 dominates) blocks the
# GPU between clients. With a persistent shared model the weights must be snapshotted
# to CPU synchronously — before the next client overwrites them via set_parameters —
# but the serialize + disk-write can then run on a background thread while the next
# client loads and trains, overlapping this client's save with the next one's I/O
# and compute. One worker: the writes are disk-bound, so a single writer avoids
# saturating the disk with parallel multi-GB writes and keeps ordering trivial. All
# state is per Ray-actor process; the executor is created lazily per process.
_SAVE_EXECUTOR = None
_SAVE_FUTURES = {}                       # path_prefix -> in-flight save Future
_SAVE_LOCK = threading.Lock()


def _save_executor():
    global _SAVE_EXECUTOR
    if _SAVE_EXECUTOR is None:
        _SAVE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="model-saver")
    return _SAVE_EXECUTOR


def _await_pending_save(path_prefix: str) -> None:
    """Block until any in-flight background save of this prefix has finished, so a
    reader (load) never sees a half-written file and a re-save never races the old
    one. A full round separates a client's save from its next load, so this is
    almost always already complete and returns at once. Re-raises write errors."""
    with _SAVE_LOCK:
        fut = _SAVE_FUTURES.get(path_prefix)
    if fut is not None:
        fut.result()


def _cpu_clone_sd(obj):
    """Deep-copy a state_dict-like structure, moving every tensor to CPU. Snapshots
    GPU state synchronously so the background writer only ever touches CPU memory
    (the live GPU tensors are overwritten/freed by the next client)."""
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _cpu_clone_sd(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_cpu_clone_sd(v) for v in obj)
    return obj


def _write_snapshot(path_prefix: str, snapshot: dict) -> None:
    for suffix, sd in snapshot.items():
        torch.save(sd, f"{path_prefix}{suffix}")

# Throughput flags — set at import so every Ray worker process picks them up.
#torch.set_float32_matmul_precision("high")
#torch.backends.cuda.matmul.allow_tf32 = True
#torch.backends.cudnn.allow_tf32 = True
#torch.backends.cudnn.benchmark = True


class SimpleBYOL:
    """
    BYOL trainer that owns all internal models and implements the full
    interface consumed by FedClient, so the client stays methodology-agnostic.

    Public interface:
        # Flower parameter exchange
        get_parameters() -> list[np.ndarray]
        set_parameters(parameters)
        num_encoder_float_params() -> int   # tells server where to split encoder/pred

        # FedEMA integration
        reset_to_global(parameters)                  # Case A: hard reset
        blend_with_global(parameters, mu) -> float   # Case B: blend + return divergence

        # Persistence
        save(path_prefix)
        load(path_prefix) -> bool

        # Training
        train(loader, epochs, server_round) -> float
    """

    def __init__(
        self,
        online_encoder: nn.Module,
        target_encoder: nn.Module,
        predictor: nn.Module,
        # p_model / p_predictor removed: they were never read by train/forward/EMA
        # (dead symmetric-BYOL leftovers), only synced + persisted. p_model was a
        # full LokiEncoder (~1 GB with the trap module) held per client for nothing.
        lr: float = 0.032,
        moving_average_decay: float = 0.99,
        use_ema: bool = True,
        local_epochs: int = 1,
        dataset_len: int = 0,
        total_rounds: int = 100,
        persist_models: bool = False,
        use_async_save: bool = False,
    ):
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        self.device = device
        self.online_device = device
        self.target_device = device
    
        # autocast below is hardcoded device_type="cuda"; only enable it on CUDA so
        # the bf16 fast path no-ops (rather than misfires) on MPS/CPU.
        self.use_amp = (device == "cuda")

        self.online_encoder = online_encoder.to(self.online_device)
        self.target_encoder = target_encoder.to(self.target_device)
        self.predictor      = predictor.to(self.online_device)

        # VRAM: the target/EMA encoder is forward-only, never sent to the server,
        # and never read by the attack (the server reads only the *online* encoder's
        # fc1), so its trap precision is free to lower. Store its trap fc1/fc2 in
        # bf16 — NOT fp16: fp16 would overflow the CSF conv key value (>65504) and
        # underflow the ~3e-9 binning weights, whereas bf16 keeps fp32's exponent
        # range. This halves the ~3 GB target trap to ~1.5 GB. The backbone stays
        # fp32 so autocast's fp32-BN policy can't hit a dtype mismatch, and the EMA
        # update / reset_to_global / save+load all cast cleanly between fp32 online
        # and bf16 target. The online fc1 (the only tensor extraction touches) is
        # untouched and bit-identical.
        if hasattr(self.target_encoder, "loki"):
            self.target_encoder.loki.fc1.to(torch.bfloat16)
            self.target_encoder.loki.fc2.to(torch.bfloat16)

        self.local_epochs  = local_epochs
        self.total_rounds  = total_rounds
        self.dataset_len   = dataset_len
        self.lr            = lr
        self.m             = moving_average_decay
        self.use_ema       = use_ema
        # When True the online/target/predictor modules are owned by a per-process
        # cache in client.py and reused (with their torch.compile graph intact)
        # every round, so release_gpu must NOT drop them — see release_gpu.
        self.persist_models = bool(persist_models)
        # When True, save() snapshots state to CPU synchronously and writes it to
        # disk on a background thread so the ~2.25 GB serialize+write overlaps the
        # next client's load+train instead of stalling the GPU. See save()/load()
        # and the module-level _SAVE_* helpers.
        self.use_async_save = bool(use_async_save)

        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # self.optimizer = torch.optim.SGD(
        #     list(self.online_encoder.parameters()) + list(self.predictor.parameters()),
        #     lr=lr,
        #     momentum=0.9,
        #     weight_decay=0.0005,
        # )
        self.optimizer = torch.optim.SGD(
            [p for p in self.online_encoder.parameters() if p.requires_grad]
            + [p for p in self.predictor.parameters() if p.requires_grad],
            lr=lr,
            momentum=0.9,
            weight_decay=0.0005,
        )

    # ------------------------------------------------------------------ #
    # Flower parameter exchange
    # ------------------------------------------------------------------ #

    def get_parameters(self) -> list:
        """Return [encoder floats, predictor floats] as numpy arrays."""
        return self._get_sd_floats(self.online_encoder) + self._get_sd_floats(self.predictor)

    def set_parameters(self, parameters: list) -> None:
        n = self._num_sd_floats(self.online_encoder)
        self._set_sd_floats(self.online_encoder, parameters[:n])
        self._set_sd_floats(self.predictor, parameters[n:])

    def num_encoder_float_params(self) -> int:
        """Count of float state-dict entries in the encoder (server uses this to split arrays)."""
        return self._num_sd_floats(self.online_encoder)

    # ------------------------------------------------------------------ #
    # FedEMA integration
    # ------------------------------------------------------------------ #

    def reset_to_global(self, parameters: list) -> None:
        """Case A: hard-reset all models to the global parameters."""
        self.set_parameters(parameters)
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())

    def blend_with_global(self, parameters: list, mu: float) -> float:
        # Fast path for mu == 1 (the autoscaler-disabled setting used here): the
        # blended result is purely the global params, which set_parameters loads
        # directly — so there's no need to clone the whole online encoder (the
        # ~2.5 GB trap fc1 clone was the blend's VRAM peak) or run the blend. We
        # only snapshot the local *conv* weights (all _conv_l2_divergence reads)
        # before the global params overwrite them. Bit-identical to the general
        # path at mu == 1.
        if mu == 1:
            local_conv = {
                f"model.{k}": v.detach().clone()
                for k, v in self.online_encoder.state_dict().items()
                if "conv" in k and "weight" in k
            }
            self.set_parameters(parameters)
            combined_global = {f"model.{k}": v
                               for k, v in self.online_encoder.state_dict().items()}
            return float(self._conv_l2_divergence(combined_global, local_conv))

        local_model_sd = self._clone_sd(self.online_encoder.state_dict())
        local_pred_sd  = self._clone_sd(self.predictor.state_dict())

        self.set_parameters(parameters)
        # global_model_sd = self._clone_sd(self.online_encoder.state_dict())
        # global_pred_sd  = self._clone_sd(self.predictor.state_dict())
        global_model_sd = self.online_encoder.state_dict()
        global_pred_sd  = self.predictor.state_dict()

        combined_global = ({f"model.{k}": v for k, v in global_model_sd.items()} |
                           {f"pred.{k}":  v for k, v in global_pred_sd.items()})
        combined_local  = ({f"model.{k}": v for k, v in local_model_sd.items()} |
                           {f"pred.{k}":  v for k, v in local_pred_sd.items()})
        div = self._conv_l2_divergence(combined_global, combined_local)

        # Blend in place into the local snapshot clones (div already computed
        # above, so mutating them is safe), then load.
        blended_model = self._blend_sds(local_model_sd, global_model_sd, mu)
        blended_pred  = self._blend_sds(local_pred_sd, global_pred_sd, mu)
        self.online_encoder.load_state_dict(blended_model)
        self.predictor.load_state_dict(blended_pred)

        return float(div)

    def divergence_from_global(self, parameters: list) -> float:
        """Paper Eq. 3 divergence ||W_g - W_k||: a single l2-norm over the WHOLE
        online encoder (sqrt of the summed squared differences across every
        encoder parameter), between the incoming global encoder and the current
        local one. Computed without mutating any model so mu can be chosen before
        blending. This matches the server's l2_norm_between and the paper, rather
        than the mean-of-per-conv-layer distance used elsewhere — the per-layer
        mean over-weights small, volatile shallow layers and trends opposite to
        the true norm."""
        keys = self._sd_float_keys(self.online_encoder)   # order matches parameters[:n_enc]
        sd   = self.online_encoder.state_dict()
        with torch.no_grad():
            sq = 0.0
            # zip stops after the encoder floats, ignoring the trailing predictor
            # arrays in `parameters` — divergence is over the encoder only.
            for k, arr in zip(keys, parameters):
                local = sd[k].detach()
                g = torch.from_numpy(arr).view_as(local).to(local.device, dtype=local.dtype)
                diff = (local - g).double()
                sq += float(diff.mul(diff).sum().item())
            return math.sqrt(sq)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path_prefix: str) -> None:
        if not self.use_async_save:
            torch.save(self.online_encoder.state_dict(), f"{path_prefix}_model.pt")
            torch.save(self.target_encoder.state_dict(), f"{path_prefix}_ema.pt")
            torch.save(self.predictor.state_dict(),      f"{path_prefix}_pred.pt")
            torch.save(self.optimizer.state_dict(),      f"{path_prefix}_optim.pt")
            return
        # Snapshot GPU -> CPU synchronously (must finish before the next client
        # overwrites the shared persistent model), then serialize+write in the
        # background so the disk I/O overlaps the next client's load+train.
        snapshot = {
            "_model.pt": _cpu_clone_sd(self.online_encoder.state_dict()),
            "_ema.pt":   _cpu_clone_sd(self.target_encoder.state_dict()),
            "_pred.pt":  _cpu_clone_sd(self.predictor.state_dict()),
            "_optim.pt": _cpu_clone_sd(self.optimizer.state_dict()),
        }
        # Finish (and surface errors from) any prior in-flight save of this prefix
        # before overwriting its files, then hand this one to the background writer.
        _await_pending_save(path_prefix)
        with _SAVE_LOCK:
            _SAVE_FUTURES[path_prefix] = _save_executor().submit(
                _write_snapshot, path_prefix, snapshot)

    def load(self, path_prefix: str) -> bool:
        """Load all state from disk. Returns True on success, False if any file is missing."""
        # A background save of this prefix from a previous round may still be in
        # flight; block until it lands so we read complete files. No-op when async
        # saving is off or the write already finished.
        _await_pending_save(path_prefix)
        try:
            on, tg = self.online_device, self.target_device
            self.online_encoder.load_state_dict(torch.load(f"{path_prefix}_model.pt",   map_location=on))
            self.target_encoder.load_state_dict(torch.load(f"{path_prefix}_ema.pt",     map_location=tg))
            self.predictor.load_state_dict(     torch.load(f"{path_prefix}_pred.pt",    map_location=on))
            # Park optimizer state (the ~2.5 GB trap-fc1 SGD momentum buffer) on the
            # CPU; train() moves it onto the GPU just before the first step(). It is
            # unused until then, so keeping it off the GPU through the blend frees
            # that VRAM at the blend's peak. See _optimizer_state_to.
            self.optimizer.load_state_dict(     torch.load(f"{path_prefix}_optim.pt",   map_location="cpu"))
            return True
        except Exception:
            return False

    def _optimizer_state_to(self, device) -> None:
        """Move the optimizer's per-parameter state (SGD momentum buffers) onto
        `device`. load() parks this state on the CPU so it doesn't occupy GPU during
        the blend; train() calls this to bring it back before any optimizer.step()."""
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device, non_blocking=True)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train(self, train_loader, epochs: int, server_round: float) -> float:
        self.online_encoder.train()
        self.predictor.train()
        # Optimizer state was parked on CPU by load() (so it didn't occupy GPU during
        # the blend); bring it onto the training device now, before any step().
        self._optimizer_state_to(self.online_device)
        total_global_steps  = self.total_rounds * self.local_epochs * self.dataset_len
        current_global_step = (server_round - 1) * self.local_epochs * self.dataset_len

        for _ in range(epochs):
            total_loss, num_batches = 0.0, 0
            for batch in train_loader:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    x1, x2 = batch
                elif isinstance(batch, (list, tuple)) and len(batch) == 3:
                    (x1, x2), _ = batch
                else:
                    raise ValueError("Expected batch to be (x1, x2) or ((x1, x2), y).")

                x1 = x1.to(self.online_device, non_blocking=True)
                x2 = x2.to(self.online_device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                    z1_online = self.online_encoder(x1, normalize=False)
                    z2_online = self.online_encoder(x2, normalize=False)
                    p1 = self.predictor(z1_online)
                    p2 = self.predictor(z2_online)
                    with torch.no_grad():
                        xt1 = x1.to(self.target_device, non_blocking=True)
                        xt2 = x2.to(self.target_device, non_blocking=True)
                        z1_target = self.target_encoder(xt1, normalize=False).detach().to(self.online_device)
                        z2_target = self.target_encoder(xt2, normalize=False).detach().to(self.online_device)
                    loss = (self._byol_loss(p1, z2_target) + self._byol_loss(p2, z1_target)).mean()

                # bf16 has sufficient dynamic range — no GradScaler needed.
                loss.backward()
                self.optimizer.step()

                current_global_step += 1
                progress = current_global_step / total_global_steps
                self.set_lr(self.lr * 0.5 * (1 + math.cos(math.pi * progress)))
                self._update_target()

                total_loss  += loss.item()
                num_batches += 1

        return total_loss / max(1, num_batches)

    def set_lr(self, lr: float) -> None:
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def release_gpu(self) -> None:
        """Drop this trainer's GPU tensors at the end of a round.

        Under Flower simulation the actor runs the clients *sequentially* and
        rebuilds a fresh trainer (build_models().to(device)) every round, so
        nothing here is reused next round. All persistent state is already on disk
        (save()) and get_parameters() has already returned CPU arrays, so the
        online/target/predictor models, their grads and the optimizer state are
        disposable once the round's result tuple is built. Releasing them here
        stops the next client's ~5 GB of models from being allocated on top of this
        trainer's still-resident copy — the footprint stacking that pushes the
        actor's high-water mark to ~2x and OOMs a later round.

        With persist_models the models are instead owned by client.py's per-process
        cache and reused every round so their torch.compile graph survives (no
        recompile). Freeing them would defeat that AND force a recompile next round,
        so we keep them resident and free only the per-round transients: the grads
        and the optimizer's momentum buffers. Because every client loads its weights
        into these SAME resident tensors (set_parameters copies in place), there is
        exactly one model set per actor — so the stacking this method guarded against
        cannot happen in the first place."""
        if self.persist_models:
            for m in (self.online_encoder, self.target_encoder, self.predictor):
                if m is not None:
                    for p in m.parameters():
                        p.grad = None
            self.optimizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return
        self.online_encoder = None
        self.target_encoder = None
        self.predictor      = None
        self.optimizer      = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _byol_loss(p, z):
        p = F.normalize(p, dim=-1, p=2)
        z = F.normalize(z, dim=-1, p=2)
        return 2 - 2 * (p * z).sum(dim=-1)

    @torch.no_grad()
    def _update_target(self) -> None:
        # EMA every online param into its target counterpart, fused across the
        # backbone's many small tensors into two foreach kernels — was a Python
        # loop launching ~2 kernels per parameter every step. torch._foreach_*
        # groups by (device, dtype) internally, so the mixed-precision target
        # (fp32 backbone + bf16 trap fc1/fc2, see __init__) and any cross-device
        # model-parallel split are handled with no special-casing of any param:
        # verified bit-identical to the old mul_/add_ path, including the bf16 add.
        # p_on (online_device) -> p_tgt (target_device): the .to() is a no-op when
        # single-device, and a cross-GPU copy under model parallelism.
        tgt = [p.data for p in self.target_encoder.parameters()]
        src = [
            p_on.data.to(p_tgt.device, non_blocking=True) if p_on.device != p_tgt.device else p_on.data
            for p_on, p_tgt in zip(self.online_encoder.parameters(), self.target_encoder.parameters())
        ]
        torch._foreach_mul_(tgt, self.m)
        torch._foreach_add_(tgt, src, alpha=1 - self.m)

    @staticmethod
    def _sd_float_keys(module: nn.Module) -> list:
        sd = module.state_dict()
        return [k for k, v in sd.items() if torch.is_tensor(v) and v.is_floating_point()]

    @staticmethod
    def _clone_sd(sd: dict) -> dict:
        return {k: v.detach().clone() for k, v in sd.items()}

    def _get_sd_floats(self, module: nn.Module) -> list:
        sd   = module.state_dict()
        keys = self._sd_float_keys(module)
        return [sd[k].detach().cpu().numpy() for k in keys]

    def _set_sd_floats(self, module: nn.Module, arrays: list) -> None:
        sd   = module.state_dict()
        keys = self._sd_float_keys(module)
        if len(arrays) != len(keys):
            raise ValueError(
                f"{module.__class__.__name__}: got {len(arrays)} arrays, expected {len(keys)}"
            )
        # Copy each array straight into the existing parameter storage instead of
        # building a whole new state_dict of fresh GPU tensors and load_state_dict'ing
        # it. The old version transiently doubled every tensor on the GPU — for the
        # ~2.5 GB trap fc1 that extra copy is exactly what OOMed set_parameters in the
        # blend path. `dst.copy_(cpu_src)` streams host->device directly into dst's
        # storage with no intermediate GPU allocation, and casts dtype as needed.
        # state_dict() tensors alias the module's params/buffers, so this updates the
        # module in place. Numerically identical — no change to FedEMA/FedBYOL.
        with torch.no_grad():
            for k, arr in zip(keys, arrays):
                dst = sd[k]
                dst.copy_(torch.from_numpy(arr).view_as(dst))

    def _num_sd_floats(self, module: nn.Module) -> int:
        return len(self._sd_float_keys(module))

    @staticmethod
    def _blend_sds(local_sd: dict, global_sd: dict, mu: float) -> dict:
        """mu=1 → pure global, mu=0 → pure local."""
        out = {}
        for k in global_sd:
            g, l = global_sd[k], local_sd[k]
            if torch.is_tensor(g) and g.is_floating_point():
                # In-place on the throwaway local clone: l*(1-mu) + g*mu. The old
                # `(1-mu)*l + mu*g` allocated three fresh tensors per key — for the
                # ~2.5 GB trap fc1 that spiked ~7 GB and OOMed the blend. mul_/add_
                # reuse l's storage and allocate nothing.
                out[k] = l.mul_(1.0 - mu).add_(g, alpha=mu)
            elif "num_batches_tracked" in k:
                out[k] = l
            else:
                out[k] = g
        return out

    def _conv_l2_divergence(self, global_sd: dict, local_sd: dict) -> float:
        """Average L2 distance over conv weight tensors — signal for the FedEMA autoscaler."""
        with torch.no_grad():
            total, count = 0.0, 0
            for name in local_sd:
                if "conv" in name and "weight" in name:
                    d = torch.dist(
                        local_sd[name].detach().view(1, -1),
                        global_sd[name].detach().view(1, -1),
                        2,
                    )
                    total += d.item()
                    count += 1
            return total / count if count > 0 else 0.0


if __name__ == "__main__":
    pass

import io
import gc
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

# Throughput flags — set at import so every Ray worker process picks them up.
#torch.set_float32_matmul_precision("high")
#torch.backends.cuda.matmul.allow_tf32 = True
#torch.backends.cudnn.allow_tf32 = True
#torch.backends.cudnn.benchmark = False


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
        model_parallel: bool = False,
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
        if model_parallel and torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            by_mem = sorted(
                range(torch.cuda.device_count()),
                key=lambda i: torch.cuda.get_device_properties(i).total_memory,
            )
            self.target_device = f"cuda:{by_mem[0]}"     # smallest GPU
            self.online_device = f"cuda:{by_mem[-1]}"    # largest GPU
        self.model_parallel = self.online_device != self.target_device
    
        self.use_amp = True

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

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path_prefix: str) -> None:
        torch.save(self.online_encoder.state_dict(), f"{path_prefix}_model.pt")
        torch.save(self.target_encoder.state_dict(), f"{path_prefix}_ema.pt")
        torch.save(self.predictor.state_dict(),      f"{path_prefix}_pred.pt")
        torch.save(self.optimizer.state_dict(),      f"{path_prefix}_optim.pt")

    def load(self, path_prefix: str) -> bool:
        """Load all state from disk. Returns True on success, False if any file is missing."""
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
        actor's high-water mark to ~2x and OOMs a later round."""
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
        # p_on (online_device) -> p_tgt (target_device): the .to() is a no-op when
        # single-device, and a cross-GPU copy under model parallelism.
        for p_on, p_tgt in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            src = p_on.data.to(p_tgt.device, non_blocking=True) if p_on.device != p_tgt.device else p_on.data
            p_tgt.data.mul_(self.m).add_(src, alpha=1 - self.m)

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

import io
import os
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

# Throughput flags — set at import so every Ray worker process picks them up.
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


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
    
        self.use_amp = False

        self.online_encoder = online_encoder.to(self.online_device)
        self.target_encoder = target_encoder.to(self.target_device)
        self.predictor      = predictor.to(self.online_device)

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

        # Blend (reads global live + local snapshot) into new tensors, then load.
        blended_model = self._blend_sds(local_model_sd, global_model_sd, mu)
        blended_pred  = self._blend_sds(local_pred_sd, global_pred_sd, mu)
        self.online_encoder.load_state_dict(blended_model)
        self.predictor.load_state_dict(blended_pred)

        return float(div)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path_prefix: str) -> None:
        # Ensure the target directory exists (e.g. local_weights/), since it is
        # gitignored and absent on a fresh checkout such as a cluster node.
        parent = os.path.dirname(path_prefix)
        if parent:
            os.makedirs(parent, exist_ok=True)
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
            self.optimizer.load_state_dict(     torch.load(f"{path_prefix}_optim.pt",   map_location=on))
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train(self, train_loader, epochs: int, server_round: float) -> float:
        self.online_encoder.train()
        self.predictor.train()
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
        new_sd = {
            k: torch.from_numpy(arr).to(sd[k].device).to(dtype=sd[k].dtype).view_as(sd[k])
            for k, arr in zip(keys, arrays)
        }
        module.load_state_dict(new_sd, strict=False)

    def _num_sd_floats(self, module: nn.Module) -> int:
        return len(self._sd_float_keys(module))

    @staticmethod
    def _blend_sds(local_sd: dict, global_sd: dict, mu: float) -> dict:
        """mu=1 → pure global, mu=0 → pure local."""
        out = {}
        for k in global_sd:
            g, l = global_sd[k], local_sd[k]
            if torch.is_tensor(g) and g.is_floating_point():
                out[k] = torch.lerp(l, g, mu)
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

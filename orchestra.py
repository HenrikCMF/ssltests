"""Orchestra trainer (Lubana et al., ICML 2022).

Numerics follow the official akhilmathurs/orchestra implementation
(sknopp, local/global clustering, clustering + rotation-degeneracy loss).
The public surface matches SimpleBYOL so FedClient stays methodology-agnostic.
The rotation head, FIFO memory, and centroids are local trainer state — they
are not Flower parameters (the online encoder is).
"""
import gc
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from BYOL import (
    _SAVE_FUTURES,
    _SAVE_LOCK,
    _await_pending_save,
    _cpu_clone_sd,
    _save_executor,
    _write_snapshot,
)
from architectures import ORCHESTRA_PROJ_DIM


def sknopp(cZ, lamd=25, max_iters=100):
    """Sinkhorn–Knopp equal-size soft assignments. Official orchestra.models.sknopp.

    cZ: [N_samples, N_centroids] cosine scores.
    Returns [N_samples, N_centroids] scaled so columns/rows meet the uniform
    marginals (each sample mass 1, each cluster mass N/K after the N_samples factor).
    """
    with torch.no_grad():
        N_samples, N_centroids = cZ.shape
        probs = F.softmax(cZ * lamd, dim=1).T  # [N_centroids, N_samples]

        r = torch.ones((N_centroids, 1), device=probs.device) / N_centroids
        c = torch.ones((N_samples, 1), device=probs.device) / N_samples

        inv_N_centroids = 1.0 / N_centroids
        inv_N_samples = 1.0 / N_samples

        err = 1e3
        for it in range(max_iters):
            r = inv_N_centroids / (probs @ c)
            c_new = inv_N_samples / (r.T @ probs).T
            if it % 10 == 0:
                err = torch.nansum(torch.abs(c / c_new - 1))
            c = c_new
            if err < 1e-2:
                break

        probs *= c.squeeze()
        probs = probs.T
        probs *= r.squeeze()
        return probs * N_samples


def pack_centroids(arr: np.ndarray) -> bytes:
    return np.ascontiguousarray(arr, dtype=np.float32).tobytes()


def unpack_centroids(blob, n: int, d: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).reshape(int(n), int(d)).copy()


def global_clustering(Z, n_global, temperature=0.1, steps=500, lr=0.01, device=None):
    """Server-side global clustering. Official orchestra.orchestra.global_clustering.

    Z: [N, D] local centroids from participating clients.
    Returns [G, D] float32 numpy, L2-normalized rows.
    """
    if device is None:
        device = torch.device("cpu")
    if not torch.is_tensor(Z):
        Z = torch.as_tensor(Z, dtype=torch.float32)
    Z1 = Z.to(device)
    D = Z1.shape[1]
    centroids = nn.Linear(D, n_global, bias=False).to(device)
    optimizer = torch.optim.SGD(
        centroids.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4,
    )
    train_loss = 0.0
    for round_idx in range(steps):
        with torch.no_grad():
            sk_assigns = sknopp(centroids(Z1))
        optimizer.zero_grad()
        probs1 = F.softmax(centroids(F.normalize(Z1, dim=1)) / temperature, dim=1)
        loss = -F.cosine_similarity(sk_assigns, probs1, dim=-1).mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            centroids.weight.copy_(F.normalize(centroids.weight.data.clone(), dim=1))
            train_loss += float(loss.item())
    print(f"[orchestra] global clustering loss={train_loss / max(1, steps):.4f} "
          f"N={int(Z1.shape[0])} G={n_global} D={D}")
    return centroids.weight.detach().cpu().numpy().astype(np.float32)


class SimpleOrchestra:
    """
    Orchestra trainer that owns the online/target encoders and implements the
    interface consumed by FedClient.

    Public interface matches SimpleBYOL, plus:
        set_global_centroids(arr [G, D])
        get_local_centroids() -> ndarray [L, D]
    """

    def __init__(
        self,
        online_encoder: nn.Module,
        target_encoder: nn.Module,
        lr: float = 0.01,
        moving_average_decay: float = 0.99,
        local_epochs: int = 1,
        dataset_len: int = 0,
        total_rounds: int = 100,
        persist_models: bool = False,
        use_async_save: bool = False,
        n_local: int = 8,
        n_global: int = 64,
        memory_size: int = 2048,
        temperature: float = 0.1,
        proj_dim: int = ORCHESTRA_PROJ_DIM,
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
        self.use_amp = (device == "cuda")

        self.online_encoder = online_encoder.to(self.online_device)
        self.target_encoder = target_encoder.to(self.target_device)

        if hasattr(self.target_encoder, "loki"):
            self.target_encoder.loki.fc1.to(torch.bfloat16)
            self.target_encoder.loki.fc2.to(torch.bfloat16)

        self.local_epochs = local_epochs
        self.total_rounds = total_rounds
        self.dataset_len = dataset_len
        self.lr = lr
        self.m = moving_average_decay
        self.persist_models = bool(persist_models)
        self.use_async_save = bool(use_async_save)

        self.n_local = int(n_local)
        self.n_global = int(n_global)
        self.m_size = int(memory_size)
        self.T = float(temperature)
        self.proj_dim = int(proj_dim)

        self.deg_layer = nn.Linear(self.proj_dim, 4).to(self.online_device)
        self.mem = torch.zeros(self.m_size, self.proj_dim, device=self.online_device)
        self.mem_count = 0
        self.global_centroids = None  # [G, D] on online_device
        self.local_centroids = torch.zeros(
            self.n_local, self.proj_dim, device=self.online_device,
        )
        self._has_global_centroids = False

        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.SGD(
            [p for p in self.online_encoder.parameters() if p.requires_grad]
            + [p for p in self.deg_layer.parameters() if p.requires_grad],
            lr=lr,
            momentum=0.9,
            weight_decay=1e-4,
        )

    # ------------------------------------------------------------------ #
    # Centroids
    # ------------------------------------------------------------------ #

    def set_global_centroids(self, arr) -> None:
        t = torch.as_tensor(arr, dtype=torch.float32, device=self.online_device)
        if t.ndim != 2 or t.shape[1] != self.proj_dim:
            raise ValueError(
                f"global centroids expected [G, {self.proj_dim}], got {tuple(t.shape)}"
            )
        self.global_centroids = t.contiguous()
        self.n_global = int(t.shape[0])
        self._has_global_centroids = True

    def get_local_centroids(self) -> np.ndarray:
        return self.local_centroids.detach().cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------ #
    # Flower parameter exchange (online encoder only)
    # ------------------------------------------------------------------ #

    def get_parameters(self) -> list:
        return self._get_sd_floats(self.online_encoder)

    def set_parameters(self, parameters: list) -> None:
        self._set_sd_floats(self.online_encoder, parameters)

    def num_encoder_float_params(self) -> int:
        return self._num_sd_floats(self.online_encoder)

    # ------------------------------------------------------------------ #
    # FedEMA integration (encoder-only; Orchestra uses mu=1 / FedAvg)
    # ------------------------------------------------------------------ #

    def reset_to_global(self, parameters: list) -> None:
        self.set_parameters(parameters)
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())

    def blend_with_global(self, parameters: list, mu: float) -> float:
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
        self.set_parameters(parameters)
        global_model_sd = self.online_encoder.state_dict()
        combined_global = {f"model.{k}": v for k, v in global_model_sd.items()}
        combined_local = {f"model.{k}": v for k, v in local_model_sd.items()}
        div = self._conv_l2_divergence(combined_global, combined_local)
        blended_model = self._blend_sds(local_model_sd, global_model_sd, mu)
        self.online_encoder.load_state_dict(blended_model)
        return float(div)

    def divergence_from_global(self, parameters: list) -> float:
        keys = self._sd_float_keys(self.online_encoder)
        sd = self.online_encoder.state_dict()
        with torch.no_grad():
            sq = 0.0
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
        extra = {
            "mem": self.mem.detach().cpu(),
            "mem_count": int(self.mem_count),
            "local_centroids": self.local_centroids.detach().cpu(),
            "global_centroids": (
                None if self.global_centroids is None
                else self.global_centroids.detach().cpu()
            ),
            "has_global": bool(self._has_global_centroids),
        }
        if not self.use_async_save:
            torch.save(self.online_encoder.state_dict(), f"{path_prefix}_model.pt")
            torch.save(self.target_encoder.state_dict(), f"{path_prefix}_ema.pt")
            torch.save(self.deg_layer.state_dict(),      f"{path_prefix}_deg.pt")
            torch.save(self.optimizer.state_dict(),      f"{path_prefix}_optim.pt")
            torch.save(extra,                            f"{path_prefix}_mem.pt")
            return
        snapshot = {
            "_model.pt": _cpu_clone_sd(self.online_encoder.state_dict()),
            "_ema.pt":   _cpu_clone_sd(self.target_encoder.state_dict()),
            "_deg.pt":   _cpu_clone_sd(self.deg_layer.state_dict()),
            "_optim.pt": _cpu_clone_sd(self.optimizer.state_dict()),
            "_mem.pt":   extra,
        }
        _await_pending_save(path_prefix)
        with _SAVE_LOCK:
            _SAVE_FUTURES[path_prefix] = _save_executor().submit(
                _write_snapshot, path_prefix, snapshot)

    def load(self, path_prefix: str) -> bool:
        _await_pending_save(path_prefix)
        try:
            on, tg = self.online_device, self.target_device
            self.online_encoder.load_state_dict(
                torch.load(f"{path_prefix}_model.pt", map_location=on))
            self.target_encoder.load_state_dict(
                torch.load(f"{path_prefix}_ema.pt", map_location=tg))
            self.deg_layer.load_state_dict(
                torch.load(f"{path_prefix}_deg.pt", map_location=on))
            self.optimizer.load_state_dict(
                torch.load(f"{path_prefix}_optim.pt", map_location="cpu"))
            extra = torch.load(f"{path_prefix}_mem.pt", map_location="cpu")
            self.mem = extra["mem"].to(self.online_device)
            self.mem_count = int(extra["mem_count"])
            self.local_centroids = extra["local_centroids"].to(self.online_device)
            g = extra.get("global_centroids")
            if g is None:
                self.global_centroids = None
                self._has_global_centroids = False
            else:
                self.global_centroids = g.to(self.online_device)
                self._has_global_centroids = bool(extra.get("has_global", True))
            return True
        except Exception:
            return False

    def _optimizer_state_to(self, device) -> None:
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device, non_blocking=True)

    # ------------------------------------------------------------------ #
    # Memory + clustering
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _bn_snapshot(self, module: nn.Module) -> dict:
        return {k: v.detach().clone() for k, v in module.state_dict().items() if "bn" in k}

    @torch.no_grad()
    def reset_memory(self, loader) -> None:
        """Fill the FIFO with target-projector features. Official reset_memory."""
        self.target_encoder.train()
        reset_dict = self._bn_snapshot(self.target_encoder)
        proj_bank = []
        n_samples = 0
        for batch in loader:
            x1 = self._unpack_x1(batch).to(self.target_device, non_blocking=True)
            z = F.normalize(self.target_encoder(x1, normalize=False), dim=1)
            proj_bank.append(z.detach())
            n_samples += z.shape[0]
            if n_samples >= self.m_size:
                break
        if proj_bank:
            bank = torch.cat(proj_bank, dim=0).contiguous()
            if bank.shape[0] > self.m_size:
                bank = bank[: self.m_size]
            self.mem.zero_()
            self.mem[: bank.shape[0]].copy_(bank.to(self.mem.device))
            self.mem_count = int(bank.shape[0])
        self.target_encoder.load_state_dict(reset_dict, strict=False)

    @torch.no_grad()
    def update_memory(self, F) -> None:
        N = F.shape[0]
        if N >= self.m_size:
            self.mem.copy_(F[-self.m_size:].detach())
            self.mem_count = self.m_size
            return
        self.mem[:-N] = self.mem[N:].detach().clone()
        self.mem[-N:] = F.detach()
        self.mem_count = min(self.m_size, self.mem_count + N)

    @torch.no_grad()
    def local_clustering(self) -> None:
        """Equal-size local centroids via Sinkhorn–Knopp. Official local_clustering."""
        n = max(self.mem_count, 0)
        if n < self.n_local:
            return
        Z = self.mem[:n].detach().clone()
        idx = np.random.choice(Z.shape[0], self.n_local, replace=False)
        centroids = Z[idx]
        for _ in range(5):
            assigns = sknopp(Z @ centroids.T, max_iters=10)
            choice_cluster = torch.argmax(assigns, dim=1)
            for index in range(self.n_local):
                selected = torch.nonzero(choice_cluster == index, as_tuple=False).squeeze()
                if selected.ndim == 0:
                    selected = selected.unsqueeze(0)
                selected = torch.index_select(Z, 0, selected)
                if selected.shape[0] == 0:
                    selected = Z[torch.randint(len(Z), (1,), device=Z.device)]
                centroids[index] = F.normalize(selected.mean(dim=0), dim=0)
        self.local_centroids = centroids.to(self.online_device)

    @staticmethod
    def _unpack_x1(batch):
        if isinstance(batch, (list, tuple)) and len(batch) == 2 and isinstance(batch[0], (list, tuple)):
            batch = batch[0]
        if isinstance(batch, (list, tuple)) and len(batch) >= 1:
            return batch[0]
        raise ValueError("Orchestra batch must start with view x1")

    @staticmethod
    def _unpack_train_batch(batch):
        # (x1, x2, x3, angle) or ((x1, x2, x3, angle), y)
        if isinstance(batch, (list, tuple)) and len(batch) == 2 and isinstance(batch[0], (list, tuple)):
            batch = batch[0]
        if isinstance(batch, (list, tuple)) and len(batch) == 4:
            return batch
        raise ValueError("Expected Orchestra batch (x1, x2, x3, angle).")

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train(self, train_loader, epochs: int, server_round: float) -> float:
        # Official ssl_train: round 0 is memory+local clustering only.
        if (not self._has_global_centroids) or int(server_round) <= 1:
            self.reset_memory(train_loader)
            self.local_clustering()
            return 0.0

        self.online_encoder.train()
        self.deg_layer.train()
        self._optimizer_state_to(self.online_device)
        C = self.global_centroids.detach()  # [G, D]

        total_loss, num_batches = 0.0, 0
        for _ in range(epochs):
            for batch in train_loader:
                x1, x2, x3, deg_labels = self._unpack_train_batch(batch)
                x1 = x1.to(self.online_device, non_blocking=True)
                x2 = x2.to(self.online_device, non_blocking=True)
                x3 = x3.to(self.online_device, non_blocking=True)
                deg_labels = deg_labels.to(self.online_device, non_blocking=True).long()
                self.optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                    z2 = F.normalize(self.online_encoder(x2, normalize=False), dim=1)
                    logp_z2 = torch.log(F.softmax((z2 @ C.T) / self.T, dim=1))
                    with torch.no_grad():
                        self._update_target()
                        xt1 = x1.to(self.target_device, non_blocking=True)
                        tz1 = F.normalize(
                            self.target_encoder(xt1, normalize=False), dim=1
                        ).to(self.online_device)
                        tp1 = F.softmax((tz1 @ C.T) / self.T, dim=1)
                    l_cluster = -(tp1 * logp_z2).sum(dim=1).mean()
                    deg_preds = self.deg_layer(self.online_encoder(x3, normalize=False))
                    l_deg = F.cross_entropy(deg_preds, deg_labels)
                    loss = l_cluster + l_deg

                loss.backward()
                self.optimizer.step()
                with torch.no_grad():
                    self.update_memory(tz1.float())

                total_loss += float(loss.item())
                num_batches += 1

        self.local_clustering()
        return total_loss / max(1, num_batches)

    def set_lr(self, lr: float) -> None:
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def release_gpu(self) -> None:
        if self.persist_models:
            for m in (self.online_encoder, self.target_encoder, self.deg_layer):
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
        self.deg_layer = None
        self.optimizer = None
        self.mem = None
        self.global_centroids = None
        self.local_centroids = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Private helpers (same as SimpleBYOL)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _update_target(self) -> None:
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
        sd = module.state_dict()
        keys = self._sd_float_keys(module)
        return [sd[k].detach().cpu().numpy() for k in keys]

    def _set_sd_floats(self, module: nn.Module, arrays: list) -> None:
        sd = module.state_dict()
        keys = self._sd_float_keys(module)
        if len(arrays) != len(keys):
            raise ValueError(
                f"{module.__class__.__name__}: got {len(arrays)} arrays, expected {len(keys)}"
            )
        with torch.no_grad():
            for k, arr in zip(keys, arrays):
                dst = sd[k]
                dst.copy_(torch.from_numpy(arr).view_as(dst))

    def _num_sd_floats(self, module: nn.Module) -> int:
        return len(self._sd_float_keys(module))

    @staticmethod
    def _blend_sds(local_sd: dict, global_sd: dict, mu: float) -> dict:
        out = {}
        for k in global_sd:
            g, l = global_sd[k], local_sd[k]
            if torch.is_tensor(g) and g.is_floating_point():
                out[k] = l.mul_(1.0 - mu).add_(g, alpha=mu)
            elif "num_batches_tracked" in k:
                out[k] = l
            else:
                out[k] = g
        return out

    def _conv_l2_divergence(self, global_sd: dict, local_sd: dict) -> float:
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

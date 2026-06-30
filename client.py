import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
import flwr as fl
from pathlib import Path
from dataloader import CIFAR10BYOLClientData
from architectures import build_models
from SSL import SimpleBYOL
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
                 loki_num_kernels: int = 3, model_parallel: bool = False):
        self.cid          = int(cid)
        self.local_epochs = local_epochs
        self.tau          = 0.7
        self.lambda_k     = None
        self.NUM_ROUND    = 0.0
        # When True the BYOL encoder carries a (gated) LOKI attack module at its
        # input; the malicious server arms it for this client only if it is the
        # attack target (see server.FedEMAStrategyWithKnn).
        self.loki             = bool(loki)
        self.loki_fc_size     = loki_fc_size
        self.loki_num_kernels = loki_num_kernels

        self._weight_prefix = f"local_weights/client{self.cid}"
        self._lambda_path   = f"local_weights/lambda{self.cid}.pkl"

        # Reuse the per-process data object across rounds (see _DATA_CACHE).
        cache_key = (self.cid, num_partitions, batch_size)
        data_obj = _DATA_CACHE.get(cache_key)
        if data_obj is None:
            # Verify/download the dataset only on the first client built in this
            # process; subsequent constructions skip the MD5 integrity check.
            global _DATASET_VERIFIED
            download = not _DATASET_VERIFIED
            _DATASET_VERIFIED = True

            data_obj = CIFAR10BYOLClientData(
                num_clients=num_partitions,
                cid=self.cid,
                classes_per_client=2,
                batch_size=batch_size,
                # 0 = load in-process (no DataLoader subprocesses). Under the
                # Flower/Ray simulation on a many-core node, forked workers
                # deadlock and spawned workers stall; single-process loading of
                # CIFAR-32x32 is cheap and GPU training dominates anyway.
                num_workers=6,
                keep_labels=False,
                data_dir="./data",
                seed=12345,
                device=DEVICE,
                download=download,
            )
            _DATA_CACHE[cache_key] = data_obj
        self.train_loader, self.val_loader = data_obj.get_loaders()

        # Linear LR scaling rule (BYOL): lr grows with batch size off a
        # reference of BYOL_REF_BATCH, so larger batches stay well-tuned.
        byol_lr = byol_base_lr * batch_size / BYOL_REF_BATCH
        model, ema, predictor = build_models(
            emb_dim=embedding_size,
            loki=self.loki,
            loki_fc_size=self.loki_fc_size,
            loki_num_kernels=self.loki_num_kernels,
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
            model_parallel=model_parallel,
        )

    # ------------------------------------------------------------------ #
    # Flower interface
    # ------------------------------------------------------------------ #

    def fit(self, parameters, config):
        selected_prev  = bool(int(config.get("selected_prev", 0)))
        self.NUM_ROUND = float(config.get("server_round", -1.0))

        has_local = self._load_local()

        if not has_local or not selected_prev:
            # Case A: first round, or not selected last round — hard reset to global.
            self.trainer.reset_to_global(parameters)
        else:
            # Case B: blend local with global; trainer returns divergence for autoscaler.
            div = self.trainer.blend_with_global(parameters, mu=1)
            if self.lambda_k is None:
                self.lambda_k = self.tau / div
            mu = 1  # min(self.lambda_k * div, 1)
            if self.cid == 0:
                print(div, mu, self.lambda_k)

        train_loss = self.trainer.train(
            self.train_loader, epochs=self.local_epochs, server_round=self.NUM_ROUND
        )
        self._save_local()
        # DIAGNOSTIC: per-client worker RSS to localize the steady host-RAM climb.
        from server import log_mem
        log_mem(f"client {self.cid} fit r{int(self.NUM_ROUND)}")
        return (
            self.trainer.get_parameters(),
            len(self.train_loader.dataset),
            # Report the partition id so the (malicious) server can map Flower's
            # opaque ClientProxy ids to partitions and target a specific client.
            {"train_loss": train_loss, "cid": self.cid},
        )

    def evaluate(self, parameters, config):
        self.trainer.set_parameters(parameters)
        loss = 1
        return float(loss), len(self.val_loader.dataset), {"val_loss": float(loss)}

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _load_local(self) -> bool:
        ok = self.trainer.load(self._weight_prefix)
        # Lambda is kept separate from model state so a missing lambda file
        # doesn't falsely report no local state (original had this bug).
        if ok:
            if self.NUM_ROUND > 2:
                try:
                    with open(self._lambda_path, "rb") as f:
                        self.lambda_k = pickle.load(f)
                except Exception:
                    self.lambda_k = None
            else:
                self.lambda_k = None
        else:
            self.lambda_k = None
        return ok

    def _save_local(self) -> None:
        self.trainer.save(self._weight_prefix)
        if self.lambda_k is not None:
            with open(self._lambda_path, "wb") as f:
                pickle.dump(self.lambda_k, f)


# Kept for backward compatibility with FedEMA_run.py (used by sd_float_arrays_bn).
def _state_dict_keys_float(module: torch.nn.Module):
    sd = module.state_dict()
    return [k for k, v in sd.items()
            if torch.is_tensor(v) and v.is_floating_point()
            and "running_mean" not in k and "running_var" not in k]


if __name__ == "__main__":
    pass

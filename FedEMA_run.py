import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # only expose big GPU
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,garbage_collection_threshold:0.8")
os.environ.setdefault("RAY_memory_monitor_refresh_ms", "0")#os.environ['RAY_memory_monitor_refresh_ms']="0"

import flwr as fl
import ray
from flwr.common import ndarrays_to_parameters
from client import FedClient, _state_dict_keys_float
import shutil
from pathlib import Path
import torch
from architectures import build_models
from flwr.common import Context
import time
from server import FedEMAStrategyWithKnn
NUM_CLIENTS        = 5#5
NUM_ROUNDS         = int(os.getenv("NUM_ROUNDS", "100"))

BATCH_SIZE         = 256
EMBEDDING_SIZE     = 2048

# ── Dataset selector ──────────────────────────────────────────────────────── #
# "cifar10"       — 32×32 RGB, 10 classes, auto-downloaded via torchvision
# "cifar100"      — 32×32 RGB, 100 classes, auto-downloaded via torchvision
# "tiny_imagenet" — 64×64 RGB, 200 classes, auto-downloaded on first run
DATASET            = "cifar10"
CLASSES_PER_CLIENT = 2#40    # non-IID shards per client (e.g. 2/10 for CIFAR-10, 20/100 for CIFAR-100, 20/200 for TinyIN)
# Fraction of the training set to use, per class, for whichever DATASET is set
# above. 1.0 = full dataset; 0.5 = half the samples but all classes preserved.
DATASET_FRACTION   = 1
# When True, every client uses ~2x-strength BYOL augmentations (crop min-scale
# halved, ColorJitter magnitudes doubled, grayscale prob doubled, and blur
# always-on for TinyImageNet). Simulates clients that deliberately harden their
# views to resist reconstruction. False = the standard BYOL augmentation recipe.
STRONG_AUG         = False
# Differential-privacy defense. When True, every client perturbs the model update
# it transmits (update = local - global) before sending it. The client is blind to
# any server-injected structure (e.g. a LOKI trap), so it applies ONE uniform rule
# to every parameter array: add Gaussian noise whose L2 norm equals DP_NOISE x that
# array's own update norm. Per-layer relative scaling is essential — a single global
# clip/noise is set by LOKI's ~1e5-scaled trap layer (~246M params) and would drive
# the real model update to zero; relative noise perturbs every layer by the same
# fraction DP_NOISE, so the honest model keeps learning while the trap channel LOKI
# reads is degraded. DP_NOISE is directly "fraction of each layer's update replaced
# by noise" (0 => none; 0.05 weak, 0.2 medium, 0.5 strong); no clip calibration
# needed. Analogous to STRONG_AUG but acts on transmitted weights, not input views.
DP_MODE            = False
DP_NOISE           = 0.1    # per-layer relative noise fraction sigma (0 => no noise)
# DP_SCHEME selects HOW the noise is scaled:
#   "per_layer" — the defense: noise L2 = DP_NOISE * ||update|| per layer (uses DP_NOISE).
#   "global"    — textbook "one absolute noise level on every weight": add
#                 N(0, DP_GLOBAL_STD^2) to every update coordinate alike (uses
#                 DP_GLOBAL_STD, ignores DP_NOISE). Meant for the NEGATIVE-RESULT sweep
#                 showing no single std both kills the leak and keeps training: sweep
#                 DP_GLOBAL_STD e.g. {3e-5,1e-4,3e-4,1e-3,3e-3} and read knn_acc (train)
#                 vs LOKI m_fired / fragment quality (leak).
#   "eps"       — the only scheme carrying a real (eps, delta)-DP guarantee, and the
#                 one to use for an epsilon sweep. Clips the whole update to DP_CLIP
#                 (the bounded-sensitivity step both legacy schemes lack) and adds the
#                 Gaussian noise that dp_accounting calibrates for DP_EPS over
#                 NUM_ROUNDS. Guarantee is CLIENT-level (one client's whole dataset)
#                 and LOCAL (holds against the malicious server itself, which is the
#                 only thing worth claiming when the aggregator is the adversary).
#                 Client-level is strictly stronger than the example-level epsilon the
#                 gradient-inversion literature usually quotes, so it upper-bounds it.
#                 Sweep: DP_EPS in {1,10,1e2,1e3,1e4,1e5,1e6} vs knn_acc (train) and
#                 LOKI m_fired / fragment quality (leak). Expect utility only at the
#                 large-eps end: local client-level DP with no subsampling
#                 amplification (fraction_fit=1.0) is unavoidably noisy, which is the
#                 standard finding, not an implementation defect.
DP_SCHEME          = os.getenv("DP_SCHEME", "per_layer")          # "eps" | "per_layer" | "global"
DP_GLOBAL_STD      = float(os.getenv("DP_GLOBAL_STD", "1e-4"))     # global-scheme absolute per-coord noise std
DP_NOISE           = float(os.getenv("DP_NOISE", str(DP_NOISE)))   # per-layer sigma (env override for sweeps)
# ── eps-scheme knobs ──────────────────────────────────────────────────────── #
# DP_EPS   target privacy budget for the WHOLE run (all NUM_ROUNDS composed), not
#          per round. Smaller = more noise. This is the sweep axis.
# DP_DELTA failure probability. Convention is delta << 1/N; at NUM_CLIENTS=5 the
#          client-level unit makes 1e-5 comfortably conservative.
# DP_CLIP  L2 bound on the transmitted update. NOT a free parameter to tune per run:
#          it must be a public constant chosen a priori (re-picking it from the
#          observed norms each round would leak the very data the noise hides). Set it
#          once from a throwaway calibration run — each round prints the pre-clip
#          |update| and the resulting clip_factor — then freeze it across the sweep so
#          every eps point differs only in noise.
DP_EPS             = float(os.getenv("DP_EPS", "10.0"))
DP_DELTA           = float(os.getenv("DP_DELTA", "1e-5"))
DP_CLIP            = float(os.getenv("DP_CLIP", "1.0"))
# DP_CALIBRATE=1 measures the update norms and transmits them UNPERTURBED, so a
# public a-priori DP_CLIP can be chosen from the printout. Needed because sigma
# scales with C, so there is no way to widen C to "see the raw norm" on a live eps
# run without also inflating the noise. A calibration run is NOT private -- it is a
# no-DP run plus logging. Keep NUM_ROUNDS at its real sweep value while calibrating:
# the LR schedule is a function of total_rounds (SSL.train), so shortening the run
# would change the very norms being measured. Just stop it after a few rounds.
DP_CALIBRATE       = os.getenv("DP_CALIBRATE", "0") == "1"
# ─────────────────────────────────────────────────────────────────────────── #

BYOL_BASE_LR     = 0.032  # LR at batch size 128; scaled linearly with batch size

# FedEMA divergence-aware autoscaler (paper Algorithm 1). When True, each client
# fuses its local model with the global one via mu = 1 - min(lambda_k*div, 1)
# (dynamic, per-client). When False, mu = 1, i.e. plain FedBYOL: a pure-global
# reset every round. Flip to False to fall back to the known-good FedBYOL path.
USE_AUTOSCALER     = False

# torch.compile the client encoders/predictor for better GPU utilization. The
# compiled models are built once per Ray-actor process and reused every round
# (client._MODEL_CACHE), so the graph is NOT rebuilt each round — only the first
# round pays the trace/autotune cost; the rest run the cached graph. Set False for
# a byte-identical, eager A/B baseline.
USE_COMPILE        = True

# Overlap the ~2.25 GB per-client model save with the next client's load+train:
# state is snapshotted to CPU synchronously, then serialized + written to disk on a
# background thread instead of stalling the GPU between clients. Results are
# unchanged (same bytes hit disk, just off the critical path). Set False to A/B.
USE_ASYNC_IO       = True

# ── Heterogeneous multi-GPU client split (e.g. 5090 + 4060 Ti) ────────────── #
# Flower applies ONE client_resources to every client and Ray's fractional-GPU
# packing is capability-blind, so "N clients on the big card, M on the small one"
# cannot be expressed via num_gpus alone. When MULTI_GPU_SPLIT is True we instead
# stop Ray from managing the GPU (num_gpus=0 + NOSET_CUDA_VISIBLE_DEVICES, so every
# actor still sees both cards) and pin each Ray-actor process to one physical GPU
# via a tiny broker actor with a per-device capacity (GPU_SPLIT_CAPS).
# torch.cuda.set_device then routes all of that actor's model/tensor work to its
# card. Set False to fall back to the plain uniform-fraction path (no pinning).
MULTI_GPU_SPLIT    = False
# Concurrent-client capacity per CUDA device index, big card first. [4, 1] = up to
# 4 clients on cuda:0 and 1 on cuda:1. Earlier devices fill first and are treated
# as HARD caps (never exceeded, to protect the big card's VRAM); any overflow
# spills to the LAST device. Confirm the printed "pinned to cuda:i (name)" lines
# match your cards and flip this if the CUDA order puts the 4060 Ti at index 0.
GPU_SPLIT_CAPS     = [4, 1]
# CPUs requested per client when splitting. With 24 cores, 4.5 lets all 5 clients
# (fraction_fit=1.0) run concurrently: floor(24 / 4.5) = 5 actor slots.
GPU_SPLIT_NUM_CPUS = 4.5
_GPU_BROKER_NAME   = "gpu_slot_broker"
_PINNED_DEVICE     = None          # per actor-process cache of the acquired device

# LOKI data-reconstruction attack (attacks.Loki, server.FedEMAStrategyWithKnn).
# When True, the malicious server prepends a trap module to the encoder, arms it
# for LOKI_TARGET_CID only each round, and reconstructs that client's images from
# its returned update. Set False for a normal (unchanged) FedBYOL run. BYOL only.
from attacks import Loki, LokiConfig
LOKI_ATTACK        = True
LOKI_TARGET_CID    = 0

LOKI_EXTRACT_ALL   = False
LOKI_LOCAL_DATASET = 10000#256#   # number of target images the trap layer aims to leak
LOKI_FC_MULT       = 4#4     # FC neurons per image (split-scaling headroom)
LOKI_CSF           = 100000.0  # high CSF clears the fp32 precision floor in the FedAVG weight delta (Eq.10, CSF^2 scaling) -- low CSF collapses the leak with fc_size=40k + small SSL gradients
LOKI_SAVE_FRAGS    = LOKI_ATTACK  # dump per-bin .pt fragment stacks for offline reconstruction
# The server-side LOKI work (arming in configure_fit, reconstruction in
# aggregate_fit) is bursty, off the training-critical path, and only ever read
# out to CPU numpy / simple tensor ops -- but build_module().to(device) would put
# the [fc_size, num_clients*C*H*W] FC1 (multi-GiB) on the *same physical GPU* the
# client actor trains on, and PyTorch's caching allocator never hands that
# reservation back. Under Flower simulation the driver (strategy) and the client
# actor share one card, so keeping LOKI on CPU here frees ~10+ GiB of GPU for the
# client without slowing the training loop. Override with LOKI_SERVER_DEVICE=cuda.
_loki_device = os.environ.get("LOKI_SERVER_DEVICE", "cpu")
# One identity mapping set per client when extracting all (full model
# inconsistency); a single set when targeting one client.
_loki_num_clients = NUM_CLIENTS if LOKI_EXTRACT_ALL else 1
LOCAL_EPOCHS       = 5# if DATASET == "cifar10" else 8
_image_shape = (3, 64, 64) if DATASET == "tiny_imagenet" else (3, 32, 32)
loki_config = LokiConfig(
    image_shape=_image_shape, num_clients=_loki_num_clients, local_dataset_size=LOKI_LOCAL_DATASET,
    fc_multiplier=LOKI_FC_MULT, csf=LOKI_CSF, fedavg=True,
    bias_mean=0.0, bias_std=0.5, device=_loki_device,
)

_loki = Loki(loki_config)
LOKI_FC_SIZE     = _loki.fc_size       # FC1 size used by the inserted module
LOKI_NUM_KERNELS = _loki.num_kernels   # one identity mapping set (= 3 for RGB)
@ray.remote(num_cpus=0)
class _GpuSlotBroker:
    """Hands each Ray-actor process a stable CUDA device index, filling devices up
    to GPU_SPLIT_CAPS (big card first). Single-threaded, so concurrent acquires are
    serialized; created once via get_if_exists by whichever actor asks first."""
    def __init__(self, caps):
        self._caps = list(caps)
        self._used = [0] * len(caps)

    def acquire(self):
        for d in range(len(self._caps)):
            if self._used[d] < self._caps[d]:
                self._used[d] += 1
                return d
        # All caps full (more actors than total capacity): spill onto the LAST
        # device rather than exceed an earlier device's hard cap and OOM it.
        d = len(self._caps) - 1
        self._used[d] += 1
        return d

    def state(self):
        return {"caps": self._caps, "used": self._used}


def _pin_gpu_for_this_actor():
    """Acquire this actor process's GPU from the broker once and make it the current
    CUDA device, so every model/tensor placed with device "cuda" lands on it. Cached
    per process (acquired on the first round, reused thereafter)."""
    global _PINNED_DEVICE
    if _PINNED_DEVICE is not None:
        return _PINNED_DEVICE
    if not torch.cuda.is_available():
        _PINNED_DEVICE = -1
        return -1
    try:
        broker = _GpuSlotBroker.options(
            name=_GPU_BROKER_NAME, get_if_exists=True, lifetime="detached",
        ).remote(GPU_SPLIT_CAPS)
        idx = int(ray.get(broker.acquire.remote()))
    except Exception as e:
        print(f"[gpu-split] broker unavailable ({e!r}); defaulting to cuda:0")
        idx = 0
    torch.cuda.set_device(idx)
    _PINNED_DEVICE = idx
    print(f"[gpu-split] actor pinned to cuda:{idx} ({torch.cuda.get_device_name(idx)})")
    return idx


def client_fn(context: Context) -> fl.client.Client:
    # Pin this actor to its assigned GPU before any model is built/moved, so the
    # persistent per-process model cache and all training land on one card.
    if MULTI_GPU_SPLIT:
        _pin_gpu_for_this_actor()
    if "partition-id" in context.node_config:
        cid = int(context.node_config["partition-id"])
    else:
        cid = int(context.node_id) % NUM_CLIENTS
    return FedClient(
        cid=cid,
        num_partitions=NUM_CLIENTS,
        local_epochs=LOCAL_EPOCHS,
        batch_size=BATCH_SIZE,
        total_rounds=NUM_ROUNDS,
        embedding_size=EMBEDDING_SIZE,
        byol_base_lr=BYOL_BASE_LR,
        loki=LOKI_ATTACK,
        loki_fc_size=LOKI_FC_SIZE,
        loki_num_kernels=LOKI_NUM_KERNELS,
        dataset=DATASET,
        classes_per_client=CLASSES_PER_CLIENT,
        data_fraction=DATASET_FRACTION,
        strong_aug=STRONG_AUG,
        dp=DP_MODE,
        dp_noise_multiplier=DP_NOISE,
        dp_scheme=DP_SCHEME,
        dp_global_std=DP_GLOBAL_STD,
        dp_eps=DP_EPS,
        dp_delta=DP_DELTA,
        dp_clip=DP_CLIP,
        dp_calibrate=DP_CALIBRATE,
        use_autoscaler=USE_AUTOSCALER,
        use_compile=USE_COMPILE,
        use_async_save=USE_ASYNC_IO,
    ).to_client()
def sd_float_arrays(module: torch.nn.Module):
    sd = module.state_dict()
    keys = [k for k, v in sd.items() if torch.is_tensor(v) and v.is_floating_point()]
    return [sd[k].detach().cpu().numpy() for k in keys]

def sd_float_arrays_bn(module):
    sd = module.state_dict()
    keys = _state_dict_keys_float(module)
    return [sd[k].detach().cpu().numpy() for k in keys]


# Build the initial global parameters from the same architecture the clients use,
# so the encoder/head split broadcast in round 1 matches each client's trainer.
_tmp, _ema, tmp_pred = build_models(
    emb_dim=EMBEDDING_SIZE, loki=LOKI_ATTACK, loki_image_shape=_image_shape,
    loki_fc_size=LOKI_FC_SIZE, loki_num_kernels=LOKI_NUM_KERNELS,
)
init_nds = sd_float_arrays(_tmp) + sd_float_arrays(tmp_pred)
initial_parameters = ndarrays_to_parameters(init_nds)

# Number of float arrays that belong to the encoder (vs predictor) — the server
# uses this to split the parameter list. Computed at module level so a short
# attack-validation harness can import it.
N_MODEL = sum(1 for k, v in _tmp.state_dict().items()
              if torch.is_tensor(v) and v.is_floating_point())

# Free the build-time temporaries that are never used again: build_models makes
# three encoders, but the server only needs _tmp (eval model + param template).
# With LOKI the unused EMA encoder carries a full ~1 GB trap module (fc1+fc2),
# and init_nds is another ~1 GB of ndarrays already serialized into
# initial_parameters — releasing both trims ~2 GB of resident host RAM.
del _ema, init_nds
#BN
#init_nds = sd_float_arrays(_tmp) + sd_float_arrays(tmp_pred)
#initial_parameters = ndarrays_to_parameters(init_nds)
if __name__ == "__main__":
    if DP_MODE and DP_SCHEME == "eps" and DP_CALIBRATE:
        print("[DP] *** CALIBRATION RUN: updates are transmitted UNPERTURBED "
              "(no clip, no noise). This run has NO privacy guarantee and must not "
              "be reported as an eps point. Read [DP-CAL] lines, set DP_CLIP, "
              "then rerun with DP_CALIBRATE=0. ***")
    elif DP_MODE and DP_SCHEME == "eps":
        # Surface the calibration before committing to a 100-round run: sigma/C
        # here is the whole story of whether this eps can possibly train.
        from dp_accounting import mu_for_eps_delta, noise_std_for_eps
        _sigma = noise_std_for_eps(DP_EPS, DP_DELTA, NUM_ROUNDS, DP_CLIP)
        print(f"[DP] client-level local DP | eps={DP_EPS:g} delta={DP_DELTA:g} "
              f"over {NUM_ROUNDS} rounds (all {NUM_CLIENTS} clients participate "
              f"every round, no subsampling amplification)")
        print(f"[DP] mu_total={mu_for_eps_delta(DP_EPS, DP_DELTA):.4g}  "
              f"clip C={DP_CLIP:g}  sensitivity=2C={2 * DP_CLIP:g}  "
              f"noise sigma={_sigma:.4g}  (sigma/C={_sigma / DP_CLIP:.4g})")
    elif DP_MODE:
        print(f'[DP] scheme="{DP_SCHEME}" has NO (eps, delta) guarantee (no clip => '
              f'unbounded sensitivity). Use DP_SCHEME=eps for the epsilon sweep.')

    strategy = FedEMAStrategyWithKnn(
        tau=0.7,
        n_model_params=N_MODEL,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        data_dir="./data",
        dataset=DATASET,
        k=200,
        temperature=0.1,
        initial_parameters=initial_parameters,
        eval_model=_tmp,
        loki_config=loki_config,
        loki_enabled=LOKI_ATTACK,
        loki_target_cid=LOKI_TARGET_CID,
        loki_extract_all=LOKI_EXTRACT_ALL,
        loki_save_fragments=LOKI_SAVE_FRAGS,
    )
    start=time.time()
    if MULTI_GPU_SPLIT:
        # Hand GPU management to us: num_gpus=0 would make Ray blank each actor's
        # CUDA_VISIBLE_DEVICES and hide every card, so NOSET tells it to leave the
        # variable alone (both cards visible) and we pick one per actor via the
        # broker + set_device. PCI_BUS_ID makes torch's device order match
        # nvidia-smi (index 0 = 5090). Set in the driver env too so the raylet the
        # actors inherit from carries it, not only the per-actor runtime_env.
        # RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 is the forward-compatible spelling of
        # the same "don't blank CUDA_VISIBLE_DEVICES at num_gpus=0" behavior (and
        # silences Ray 2.53's FutureWarning about it); NOSET keeps it working today.
        _split_env = {
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        }
        os.environ.update(_split_env)
        _client_resources = {"num_cpus": GPU_SPLIT_NUM_CPUS, "num_gpus": 0}
        _ray_init_args = {
            "_temp_dir": "/home/henrik/ray_tmp",
            "runtime_env": {"env_vars": dict(_split_env)},
        }
    else:
        _client_resources = {"num_cpus": 5, "num_gpus": 0.3}
        # /tmp is a 31G RAM-backed tmpfs with a per-user quota; keep Ray's logs +
        # object spilling on the real 1.5T disk instead.
        _ray_init_args = {"_temp_dir": "/home/henrik/ray_tmp"}
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources=_client_resources,
        ray_init_args=_ray_init_args,
    )
    print(f"Total training time: {time.time()-start:.2f} seconds")
    folder = Path("local_weights")

    for p in folder.iterdir():
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()

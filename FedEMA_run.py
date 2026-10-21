import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # only expose big GPU; dp_eps_sweep.py overrides to run a point on the second card
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,garbage_collection_threshold:0.8")
os.environ.setdefault("RAY_memory_monitor_refresh_ms", "0")#os.environ['RAY_memory_monitor_refresh_ms']="0"

import flwr as fl
from flwr.common import ndarrays_to_parameters
from client import FedClient, _state_dict_keys_float, LOCAL_WEIGHTS
from architectures import ORCHESTRA_PROJ_DIM
import shutil
from pathlib import Path
import torch
from architectures import build_models
from flwr.common import Context
import time
from server import FedEMAStrategyWithKnn
from robust_aggregate import DistributionAggregator, RandomAggregator
NUM_CLIENTS        = 5#5
# NUM_ROUNDS is the run's NOMINAL horizon: it sets the BYOL LR schedule
# (BYOL.train: total_global_steps = total_rounds * local_epochs * dataset_len) AND
# the number of DP compositions sigma is calibrated for (dp_accounting, sqrt(T)).
# It is NOT the number of rounds actually executed — see SIM_ROUNDS.
NUM_ROUNDS         = int(os.getenv("NUM_ROUNDS", "100"))
# Rounds actually executed before the simulation stops. Defaults to NUM_ROUNDS
# (unchanged behaviour). Setting SIM_ROUNDS < NUM_ROUNDS runs a TRUNCATED version
# of the nominal run: the LR schedule and sigma both stay those of a NUM_ROUNDS
# run, and it simply halts early. That is the only way to get a short run whose
# early rounds are comparable to a long one's — shortening NUM_ROUNDS instead
# would compress the LR decay and shrink sigma by sqrt(T), changing two things at
# once. Used by dp_clip_sweep.py (10 executed rounds of a 100-round schedule).
SIM_ROUNDS         = int(os.getenv("SIM_ROUNDS", str(NUM_ROUNDS)))

BATCH_SIZE         = 256
EMBEDDING_SIZE     = 2048

# ── Dataset selector ──────────────────────────────────────────────────────── #
# "cifar10"       — 32×32 RGB, 10 classes, auto-downloaded via torchvision
# "cifar100"      — 32×32 RGB, 100 classes, auto-downloaded via torchvision
# DATASET overrides the default. CLASSES_PER_CLIENT follows the dataset
# (2/10, 20/100, 40/200) unless CLASSES_PER_CLIENT is set in the environment.
DATASET            = os.getenv("DATASET", "cifar10")
_CLS_DEFAULT       = {"tiny_imagenet": 40, "cifar100": 20}.get(DATASET, 2)
CLASSES_PER_CLIENT = int(os.getenv("CLASSES_PER_CLIENT", str(_CLS_DEFAULT)))
# Fraction of the training set to use, per class, for whichever DATASET is set
# above. 1.0 = full dataset; 0.5 = half the samples but all classes preserved.
DATASET_FRACTION   = 1
# When True, every client uses ~2x-strength BYOL augmentations
STRONG_AUG         = False
# Differential-privacy defense. When True, every client perturbs the model update
# it transmits (update = local - global) before sending it: the whole update is
# clipped to DP_CLIP (the bounded-sensitivity step, without which no epsilon
# exists) and the Gaussian noise dp_accounting calibrates for DP_EPS over
# NUM_ROUNDS is added. The guarantee is CLIENT-level (one client's whole dataset)
# and LOCAL (holds against the malicious server itself, which is the only thing
# worth claiming when the aggregator is the adversary). Client-level is strictly
# stronger than the example-level epsilon the gradient-inversion literature usually
# quotes, so it upper-bounds it. Sweep: DP_EPS in {1,10,1e2,1e3,1e4,1e5,1e6} vs
# knn_acc (train) and LOKI m_fired / fragment quality (leak). Expect utility only at
# the large-eps end: local client-level DP with no subsampling amplification
# (fraction_fit=1.0) is unavoidably noisy, which is the standard finding, not an
# implementation defect. Analogous to STRONG_AUG but acts on transmitted weights,
# not input views.
DP_MODE            = os.getenv("DP_MODE", "0") == "1"
# ── DP knobs ──────────────────────────────────────────────────────────────── #
# DP_CLIP  L2 bound on the transmitted update. NOT a free parameter to tune per run:
#          it must be a public constant chosen a priori (re-picking it from the
#          observed norms each round would leak the very data the noise hides). Set it
#          once from a throwaway calibration run — each round prints the pre-clip
#          |update| and the resulting clip_factor — then freeze it across the sweep so
#          every eps point differs only in noise.
DP_EPS             = float(os.getenv("DP_EPS", "10.0")) #default 1e24
DP_DELTA           = float(os.getenv("DP_DELTA", "1e-5")) #1e-5
DP_CLIP            = float(os.getenv("DP_CLIP", "1e6")) #1e6
# DP_CALIBRATE=1 measures the update norms and transmits them UNPERTURBED, so a
# public a-priori DP_CLIP can be chosen from the printout. Needed because sigma
# scales with C, so there is no way to widen C to "see the raw norm" on a live eps
# run without also inflating the noise. A calibration run is NOT private -- it is a
# no-DP run plus logging. Keep NUM_ROUNDS at its real sweep value while calibrating:
# the LR schedule is a function of total_rounds (BYOL.train), so shortening the run
# would change the very norms being measured. Just stop it after a few rounds.
DP_CALIBRATE       = os.getenv("DP_CALIBRATE", "0") == "1" #default 0
# ──────────────────────────────────────────────────────────────────── ─────── #

BYOL_BASE_LR     = 0.032  # LR at batch size 128; scaled linearly with batch size

# FedSSL methodology: "byol" (FedBYOL / FedEMA via USE_AUTOSCALER) or "orchestra".
SSL_METHOD         = os.getenv("SSL_METHOD", "byol")

# Orchestra HPs (ignored unless SSL_METHOD="orchestra"). LR is the paper
# cross-silo value at batch 256, scaled linearly with batch size.
# G/L paper defaults (CIFAR-10: 64/8, CIFAR-100: 128/16). Linear probe is
# flat across a wide range (Orchestra Table 2); regrouping of Loki fragments
# may not be -- finer G/L shrinks the prototype grain the encoder is trained
# to glue together. Env-overridable; keep G ≲ NUM_CLIENTS*L (server only
# sees that many local centroids). Tag the run so fragments/eval_model.pth
# of the G=64 L=8 baseline are not overwritten:
#   SSL_METHOD=orchestra ORCHESTRA_G=256 ORCHESTRA_L=64 RUN_TAG=_g256_l64 python FedEMA_run.py
ORCHESTRA_BASE_LR  = 0.01
ORCHESTRA_REF_BATCH = 256
ORCHESTRA_EMA      = 0.99
ORCHESTRA_T        = 0.1
_G_DEFAULT, _L_DEFAULT = (128, 16) if DATASET == "cifar100" else (64, 8)
ORCHESTRA_G        = int(os.getenv("ORCHESTRA_G", str(_G_DEFAULT)))
ORCHESTRA_L        = int(os.getenv("ORCHESTRA_L", str(_L_DEFAULT)))
ORCHESTRA_MEMORY   = 8 * BATCH_SIZE  # last 8 iterations, matching the paper

# FedEMA divergence-aware autoscaler
USE_AUTOSCALER     = False

# Server-side weight aggregation. False = Flower FedAvg (unchanged).
# True = RandomAggregator: each round draws uniformly from trimmed mean,
# winsorized mean, Huber M-estimator (no FedAvg; a linear round leaks).
USE_RANDOM_AGG     = False
# True = DistributionAggregator (robust_aggregate.py default tail 0.35).
# USE_DISTRIBUTION_AGG=1 turns it on. Unset or 0 is Flower FedAvg.
# Enable only one of USE_RANDOM_AGG and USE_DISTRIBUTION_AGG.
USE_DISTRIBUTION_AGG = os.getenv("USE_DISTRIBUTION_AGG", "0") == "1"

# torch.compile
USE_COMPILE        = True

# Overlap the ~2.25 GB per-client model save with the next client's load+train
USE_ASYNC_IO       = True

# LOKI data-reconstruction attack 
from attacks import Loki, LokiConfig
LOKI_ATTACK        = True
LOKI_TARGET_CID    = 0

LOKI_EXTRACT_ALL   = False
# "individual"       — reconstruct from the target client's own update.
# "secagg"           — aggregate first, reconstruct only from the reduced FC1.
#                      Disjoint identity sets when not arm-target-only.
# "secagg_symmetric" — Sec. 4.4 broadcast: every client gets the same module
#                      (one identity set). Reconstruct from the reduced FC1.
#                      LOKI_CSF is multiplied by NUM_CLIENTS. EXTRACT_ALL and
#                      SACRIFICE send different weights per client, so they
#                      are ignored. Independent of the two aggregator flags.
LOKI_MODE          = os.getenv("LOKI_MODE", "individual")  # "individual" | "secagg" | "secagg_symmetric"
# secagg download: False = paper attack (every client gets its own identity set).
# True = only LOKI_TARGET_CID is armed; the other clients get _inert_module so
# they train the honest residual. Reconstruction is armed - n*agg (others
# return zeros on the trap). Ignored in individual mode (already target-only).
# LOKI_EXTRACT_ALL still wins and arms everyone. Env LOKI_ARM_TARGET_ONLY=1
# overrides the bool.
LOKI_ARM_TARGET_ONLY =True
#LOKI_ARM_TARGET_ONLY =  os.getenv("LOKI_ARM_TARGET_ONLY", str(int(LOKI_ARM_TARGET_ONLY))) == "1"
# Inert-trap decoys (zero conv/bias/fc2, FC1 filled with a constant) so that
# client's images do not overwrite the constant. Reconstruction stays
# armed - n*agg on the target-only path.
#   None  — no decoy.
#   int   — that one client gets +C. Trimmed mean (K=5, drop 1/side) treats
#           +C as the high outlier and keeps a positive Δ0.
#   "all" — every client except LOKI_TARGET_CID. Lower sorted ids get -C,
#           upper get +C. K=5 uploads [-C, -C, target_weight, C, C]: while
#           |target_weight| < C the coordinate-wise median is the target, and
#           ±C cancel in a mean.
#   [ids] — same -C/+C split over that subset. A one-element list is +C.
LOKI_SACRIFICE_CID = None  # e.g. 1, or "all"
LOKI_SACRIFICE_DECOY = 100.0
LOKI_LOCAL_DATASET = int(os.getenv("TRAP_SIZE", "10000"))#256#   # number of target images the trap layer aims to leak
LOKI_FC_MULT       = 4#4     # FC neurons per image (split-scaling headroom)
LOKI_CSF           = float(os.getenv("LOKI_CSF", "100000.0"))#1e5 clears dp  # high CSF clears the fp32 precision floor in the FedAVG weight delta (Eq.10, CSF^2 scaling) -- low CSF collapses the leak with fc_size=40k + small SSL gradients
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
# One identity mapping set per client when every participant is armed
# with a distinct set (extract-all, or secagg without ARM_TARGET_ONLY).
# secagg_symmetric shares one set, same width as target-only.
_loki_symmetric = LOKI_MODE == "secagg_symmetric"
_loki_arm_all = (not _loki_symmetric) and (
    LOKI_EXTRACT_ALL or (LOKI_MODE == "secagg" and not LOKI_ARM_TARGET_ONLY))
_loki_num_clients = NUM_CLIENTS if _loki_arm_all else 1
# Fig. 6: no model inconsistency uses CSF * num_clients. FC width stays
# TRAP_SIZE * LOKI_FC_MULT; only the key value changes.
_loki_csf = LOKI_CSF * NUM_CLIENTS if _loki_symmetric else LOKI_CSF
_loki_extract_all = False if _loki_symmetric else LOKI_EXTRACT_ALL
_loki_arm_target_only = False if _loki_symmetric else LOKI_ARM_TARGET_ONLY
_loki_sacrifice_cid = None if _loki_symmetric else LOKI_SACRIFICE_CID
LOCAL_EPOCHS       = 5# if DATASET == "cifar10" else 8
_image_shape = (3, 64, 64) if DATASET == "tiny_imagenet" else (3, 32, 32)
loki_config = LokiConfig(
    image_shape=_image_shape, num_clients=_loki_num_clients, local_dataset_size=LOKI_LOCAL_DATASET,
    fc_multiplier=LOKI_FC_MULT, csf=_loki_csf, fedavg=True,
    bias_mean=0.0, bias_std=0.5, device=_loki_device,
)

_loki = Loki(loki_config)
LOKI_FC_SIZE     = _loki.fc_size       # FC1 size used by the inserted module
LOKI_NUM_KERNELS = _loki.num_kernels   # one identity mapping set (= 3 for RGB)


def client_fn(context: Context) -> fl.client.Client:
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
        dp_eps=DP_EPS,
        dp_delta=DP_DELTA,
        dp_clip=DP_CLIP,
        dp_calibrate=DP_CALIBRATE,
        use_autoscaler=USE_AUTOSCALER,
        use_compile=USE_COMPILE,
        use_async_save=USE_ASYNC_IO,
        ssl_method=SSL_METHOD,
        orchestra_base_lr=ORCHESTRA_BASE_LR,
        orchestra_n_local=ORCHESTRA_L,
        orchestra_n_global=ORCHESTRA_G,
        orchestra_memory_size=ORCHESTRA_MEMORY,
        orchestra_temperature=ORCHESTRA_T,
        orchestra_ema=ORCHESTRA_EMA,
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
    ssl_method=SSL_METHOD,
)
if tmp_pred is None:
    init_nds = sd_float_arrays(_tmp)
else:
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
del _ema, tmp_pred, init_nds
#BN
#init_nds = sd_float_arrays(_tmp) + sd_float_arrays(tmp_pred)
#initial_parameters = ndarrays_to_parameters(init_nds)
if __name__ == "__main__":
    if LOKI_MODE not in ("individual", "secagg", "secagg_symmetric"):
        raise ValueError(
            f"LOKI_MODE must be 'individual', 'secagg', or 'secagg_symmetric', "
            f"got {LOKI_MODE!r}")
    if SSL_METHOD not in ("byol", "orchestra"):
        raise ValueError(f"SSL_METHOD must be 'byol' or 'orchestra', got {SSL_METHOD!r}")
    if DATASET not in ("cifar10", "cifar100", "tiny_imagenet"):
        raise ValueError(f"DATASET must be cifar10, cifar100, or tiny_imagenet, got {DATASET!r}")
    if SSL_METHOD == "orchestra" and DATASET == "tiny_imagenet":
        raise ValueError("Orchestra is not configured for Tiny ImageNet in this run.")
    if SSL_METHOD == "orchestra":
        if ORCHESTRA_L < 2 or ORCHESTRA_G < 2:
            raise ValueError(f"ORCHESTRA_G/L must be >= 2, got G={ORCHESTRA_G} L={ORCHESTRA_L}")
        if ORCHESTRA_L > ORCHESTRA_MEMORY:
            raise ValueError(
                f"ORCHESTRA_L={ORCHESTRA_L} exceeds memory bank {ORCHESTRA_MEMORY}"
            )
        _n_seen = NUM_CLIENTS * ORCHESTRA_L
        if ORCHESTRA_G > _n_seen:
            print(f"[orchestra] warning: G={ORCHESTRA_G} > {NUM_CLIENTS}*L={_n_seen} "
                  f"local centroids the server will see; global clustering is "
                  f"underdetermined. Raise L or drop G.")
    print(f"[data] dataset={DATASET} classes_per_client={CLASSES_PER_CLIENT} "
          f"fraction={DATASET_FRACTION}")
    print(f"[SSL] method={SSL_METHOD}"
          + (f"  G={ORCHESTRA_G} L={ORCHESTRA_L} D={ORCHESTRA_PROJ_DIM} "
             f"mem={ORCHESTRA_MEMORY} lr={ORCHESTRA_BASE_LR}*{BATCH_SIZE}/{ORCHESTRA_REF_BATCH} "
             f"ema={ORCHESTRA_EMA} T={ORCHESTRA_T}"
             if SSL_METHOD == "orchestra"
             else f"  autoscaler={USE_AUTOSCALER} lr={BYOL_BASE_LR}*{BATCH_SIZE}/{128}"))
    if LOKI_ATTACK:
        if _loki_symmetric and LOKI_EXTRACT_ALL:
            print("[LOKI] secagg_symmetric ignores LOKI_EXTRACT_ALL "
                  "(disjoint identity sets are different models)")
        if _loki_symmetric and LOKI_SACRIFICE_CID is not None:
            print("[LOKI] secagg_symmetric ignores LOKI_SACRIFICE_CID "
                  "(decoys are a different model per client)")
        _csf_note = (f" csf={_loki_csf:g} ({LOKI_CSF:g}*{NUM_CLIENTS})"
                     if _loki_symmetric else "")
        print(f"[LOKI] mode={LOKI_MODE} extract_all={_loki_extract_all} "
              f"arm_target_only={_loki_arm_target_only} "
              f"sacrifice={_loki_sacrifice_cid} decoy={LOKI_SACRIFICE_DECOY:g} "
              f"random_agg={USE_RANDOM_AGG} "
              f"distribution_agg={USE_DISTRIBUTION_AGG}{_csf_note}")
    if DP_MODE and DP_CALIBRATE:
        print("[DP] *** CALIBRATION RUN: updates are transmitted UNPERTURBED "
              "(no clip, no noise). This run has NO privacy guarantee and must not "
              "be reported as an eps point. Read [DP-CAL] lines, set DP_CLIP, "
              "then rerun with DP_CALIBRATE=0. ***")
    elif DP_MODE:
        # Surface the calibration before committing to a 100-round run: sigma/C
        # here is the whole story of whether this eps can possibly train.
        from dp_accounting import mu_for_eps_delta, noise_std_for_eps
        _sigma = noise_std_for_eps(DP_EPS, DP_DELTA, NUM_ROUNDS, DP_CLIP)
        print(f"[DP] client-level local DP | eps={DP_EPS:g} delta={DP_DELTA:g} "
              f"over {NUM_ROUNDS} rounds (all {NUM_CLIENTS} clients participate "
              f"every round, no subsampling amplification)")
        if SIM_ROUNDS != NUM_ROUNDS:
            # Truncation is conservative for privacy (fewer releases than the
            # budget paid for) but must be stated: the eps above is the one for
            # the FULL horizon, and the LR schedule is the full one too.
            print(f"[DP] TRUNCATED RUN: executing {SIM_ROUNDS} of {NUM_ROUNDS} "
                  f"nominal rounds. sigma and the LR schedule are those of the "
                  f"{NUM_ROUNDS}-round run; eps={DP_EPS:g} is the budget for the "
                  f"full horizon, so {SIM_ROUNDS} rounds spend strictly less.")
        print(f"[DP] mu_total={mu_for_eps_delta(DP_EPS, DP_DELTA):.4g}  "
              f"clip C={DP_CLIP:g}  sensitivity=2C={2 * DP_CLIP:g}  "
              f"noise sigma={_sigma:.4g}  (sigma/C={_sigma / DP_CLIP:.4g})")

    if USE_RANDOM_AGG and USE_DISTRIBUTION_AGG:
        raise ValueError("enable only one of USE_RANDOM_AGG and USE_DISTRIBUTION_AGG")
    if USE_RANDOM_AGG:
        _aggregator = RandomAggregator()
    elif USE_DISTRIBUTION_AGG:
        _aggregator = DistributionAggregator()
    else:
        _aggregator = None

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
        loki_extract_all=_loki_extract_all,
        loki_secagg=(LOKI_MODE == "secagg" or _loki_symmetric),
        loki_arm_target_only=_loki_arm_target_only,
        loki_symmetric=_loki_symmetric,
        loki_sacrifice_cid=_loki_sacrifice_cid,
        loki_sacrifice_decoy=LOKI_SACRIFICE_DECOY,
        loki_save_fragments=LOKI_SAVE_FRAGS,
        orchestra_enabled=(SSL_METHOD == "orchestra"),
        orchestra_n_global=ORCHESTRA_G,
        orchestra_temperature=ORCHESTRA_T,
        aggregator=_aggregator,
    )
    # Clients torch.save into this dir without creating it (trainer.save).
    Path(LOCAL_WEIGHTS).mkdir(exist_ok=True)
    start=time.time()
    # num_gpus is the FRACTION of one GPU a client reserves, so it sets how many
    # clients Ray runs concurrently: 0.3 => 3 at once on the 32G 5090. The 16G
    # 4060 Ti must be given 1.0 (one client at a time) or it OOMs on VRAM.
    _client_resources = {"num_cpus": int(os.getenv("CLIENT_NUM_CPUS", "4")),
                         "num_gpus": float(os.getenv("CLIENT_NUM_GPUS", "0.3"))}
    # /tmp is a 31G RAM-backed tmpfs with a per-user quota; keep Ray's logs +
    # object spilling on the real 1.5T disk instead.
    _ray_init_args = {"_temp_dir": "/home/henrik/ray_tmp"}
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=SIM_ROUNDS),
        strategy=strategy,
        client_resources=_client_resources,
        ray_init_args=_ray_init_args,
    )
    print(f"Total training time: {time.time()-start:.2f} seconds")
    # Only this run's own dir -- a concurrent run on the other GPU has its own,
    # and wiping the shared name would delete its weights mid-round.
    folder = Path(LOCAL_WEIGHTS)
    if folder.is_dir():
        for p in folder.iterdir():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()

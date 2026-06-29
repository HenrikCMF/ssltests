import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # only expose big GPU
import flwr as fl
from flwr.common import ndarrays_to_parameters
from client import FedClient, _state_dict_keys_float
import shutil
from pathlib import Path
import torch
from architectures import build_models
from flwr.common import Context
import time
from server import FedEMAStrategyWithKnn
NUM_CLIENTS   = 5
NUM_ROUNDS    = 100
LOCAL_EPOCHS  = 5
BATCH_SIZE    = 128#128
EMBEDDING_SIZE=2048

BYOL_BASE_LR     = 0.032  # LR at batch size 128; scaled linearly with batch size

# LOKI data-reconstruction attack (attacks.Loki, server.FedEMAStrategyWithKnn).
# When True, the malicious server prepends a trap module to the encoder, arms it
# for LOKI_TARGET_CID only each round, and reconstructs that client's images from
# its returned update. Set False for a normal (unchanged) FedBYOL run. BYOL only.
from attacks import Loki, LokiConfig
LOKI_ATTACK        = True
LOKI_TARGET_CID    = 0
# Extraction scope. False: arm + reconstruct LOKI_TARGET_CID only (one identity
# mapping set). True: full model inconsistency (paper Sec. 3.5/4.2) -- every
# client gets its OWN identity mapping set, the per-client updates are securely
# aggregated (summed), and the server de-aggregates and reconstructs EVERY client
# from disjoint weight blocks. Fragments are written per client under
# fragments/client_XX/. NOTE: the conv layer and FC1's input grow ~linearly with
# NUM_CLIENTS, so both model memory and the per-client serialized payload scale
# with the client count -- reduce LOKI_LOCAL_DATASET when testing all-client mode.
LOKI_EXTRACT_ALL   = True
LOKI_LOCAL_DATASET = 10000#256#   # number of target images the trap layer aims to leak
LOKI_FC_MULT       = 1#4     # FC neurons per image (split-scaling headroom)
LOKI_CSF           = 100000.0  # high CSF clears the fp32 precision floor in the FedAVG weight delta (Eq.10, CSF^2 scaling) -- low CSF collapses the leak with fc_size=40k + small SSL gradients
LOKI_SAVE_FRAGS    = True  # dump per-bin .pt fragment stacks for offline reconstruction
# Model parallelism for the heavy all-client trap: each client (run sequentially)
# gets BOTH GPUs during local training -- the trainable online encoder + predictor
# on the larger GPU, the frozen EMA target encoder on the smaller one. Defaults on
# with LOKI_EXTRACT_ALL (where fc1 is widest); harmless on a single GPU.
LOKI_MODEL_PARALLEL = False#LOKI_EXTRACT_ALL
_loki_device = "cuda" if torch.cuda.is_available() else "cpu"
# One identity mapping set per client when extracting all (full model
# inconsistency); a single set when targeting one client.
_loki_num_clients = NUM_CLIENTS if LOKI_EXTRACT_ALL else 1
loki_config = LokiConfig(
    image_shape=(3, 32, 32), num_clients=_loki_num_clients, local_dataset_size=LOKI_LOCAL_DATASET,
    fc_multiplier=LOKI_FC_MULT, csf=LOKI_CSF, fedavg=True,
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
        model_parallel=LOKI_MODEL_PARALLEL,
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
    emb_dim=EMBEDDING_SIZE, loki=LOKI_ATTACK,
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
    strategy = FedEMAStrategyWithKnn(
        tau=0.7,
        n_model_params=N_MODEL,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        data_dir="./data",
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
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources = {"num_cpus": 10, "num_gpus": 0.8}
    )
    print(f"Total training time: {time.time()-start:.2f} seconds")
    folder = Path("local_weights")

    for p in folder.iterdir():
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()

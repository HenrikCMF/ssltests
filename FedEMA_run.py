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
BATCH_SIZE    = 128
EMBEDDING_SIZE=2048
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
        embedding_size=EMBEDDING_SIZE
    ).to_client()
def sd_float_arrays(module: torch.nn.Module):
    sd = module.state_dict()
    keys = [k for k, v in sd.items() if torch.is_tensor(v) and v.is_floating_point()]
    return [sd[k].detach().cpu().numpy() for k in keys]

def sd_float_arrays_bn(module):
    sd = module.state_dict()
    keys = _state_dict_keys_float(module)
    return [sd[k].detach().cpu().numpy() for k in keys]


_tmp, _ema, tmp_pred, _, _ = build_models(emb_dim=EMBEDDING_SIZE)
init_nds = sd_float_arrays(_tmp) + sd_float_arrays(tmp_pred)
initial_parameters = ndarrays_to_parameters(init_nds)
#BN
#init_nds = sd_float_arrays(_tmp) + sd_float_arrays(tmp_pred)
#initial_parameters = ndarrays_to_parameters(init_nds)
if __name__ == "__main__":
    _sd = _tmp.state_dict()
    #{f"model.{k}": v for k, v in global_model_sd.items()} | {f"pred.{k}": v for k, v in global_pred_sd.items()}
    N_MODEL = sum(1 for k, v in _sd.items() if torch.is_tensor(v) and v.is_floating_point())
    #keys = _state_dict_keys_float(_tmp)
    #N_MODEL = len(keys)     
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
        eval_model=_tmp
    )
    start=time.time()
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources = {"num_cpus": 8, "num_gpus": 0.9}
    )
    print(f"Total training time: {time.time()-start:.2f} seconds")
    folder = Path("local_weights")

    for p in folder.iterdir():
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # only expose big GPU
import flwr as fl
from client import FedClient
import shutil
from pathlib import Path
import torch
from eval import FedAvgWithKnnEval
from flwr.common import Context
import time
from server import FedEMAStrategyWithKnn
from architectures import ResNet18Projv2
NUM_CLIENTS   = 5      # how many simulated FL clients
NUM_ROUNDS    = 20     # how many FedAvg rounds
LOCAL_EPOCHS  = 5      # BYOL epochs per round per client
BATCH_SIZE    = 128

def client_fn(context: Context) -> fl.client.Client:
    if "partition-id" in context.node_config:
        cid = int(context.node_config["partition-id"])
    else:
        # Fallback only; node_id is not guaranteed to be 0..N-1
        cid = int(context.node_id) % NUM_CLIENTS

    return FedClient(
        cid=cid,
        num_partitions=NUM_CLIENTS,
        local_epochs=LOCAL_EPOCHS,
        batch_size=BATCH_SIZE,
    ).to_client()

if __name__ == "__main__":
    #strategy = FedAvgWithKnnEval(
    #    fraction_fit=1.0,
    #    fraction_evaluate=0.0,
    #    min_fit_clients=NUM_CLIENTS,
    #    min_available_clients=NUM_CLIENTS,
    #    data_dir="./data",
    #    device="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    #)
    _tmp = ResNet18Projv2(emb_dim=2048)
    N_MODEL = sum(1 for _ in _tmp.parameters())

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
    )
    start=time.time()
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources = {"num_cpus": 4, "num_gpus": 0.18}
    )
    print(f"Total training time: {time.time()-start:.2f} seconds")
    folder = Path("local_weights")

    for p in folder.iterdir():
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()

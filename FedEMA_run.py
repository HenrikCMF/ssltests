import flwr as fl
from client import FedClient
import shutil
from pathlib import Path
import torch
from eval import FedAvgWithKnnEval
NUM_CLIENTS   = 5      # how many simulated FL clients
NUM_ROUNDS    = 10      # how many FedAvg rounds
LOCAL_EPOCHS  = 1      # BYOL epochs per round per client
BATCH_SIZE    = 128

def client_fn(cid: str) -> fl.client.NumPyClient:
    """Construct a single client instance."""
    return FedClient(
        cid=int(cid),
        num_partitions=NUM_CLIENTS,
        local_epochs=LOCAL_EPOCHS,
        batch_size=BATCH_SIZE,
    )

if __name__ == "__main__":
    strategy = FedAvgWithKnnEval(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        data_dir="./data",
        device="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    folder = Path("local_weights")

    for p in folder.iterdir():
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()

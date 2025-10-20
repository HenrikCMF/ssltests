# run_sim.py
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import flwr as fl
from model import Net
from client import FlowerClient
from utils import mseFedAvg
from pathlib import Path
from server import server
from flwr.common import parameters_to_ndarrays
DATA_DIR = Path(__file__).resolve().parent / "data"   # absolute path
#artificial data scarcity
NUM_CLIENTS = 100
POP=NUM_CLIENTS+1
def client_fn(cid: str) -> FlowerClient:
    return FlowerClient(int(cid), num_partitions=POP, local_epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE).to_client()

def fit_config_factory(pre_rounds: int):
    def fit_config(server_round: int):
        return {
            "phase": server_round if server_round <= pre_rounds else "train",
            "server_round": server_round,  # handy for client logging
        }
    return fit_config

# ===== Run simulation =====

TOPK=10 #8=65%
LOCAL_EPOCHS = 1#min(int(NUM_CLIENTS/TOPK),1)*2
BATCH_SIZE = 32
ROUNDS = 3
sel_rounds=3
if __name__ == "__main__":
    base=server(NUM_CLIENTS, BATCH_SIZE, 40)
    base.fit()
    init_params = fl.common.ndarrays_to_parameters(FlowerClient.get_weights(Net()))
    strategy = mseFedAvg(
        topk=TOPK,                     # make sure TOPK <= NUM_CLIENTS
        selection_rounds=sel_rounds,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,   # ensure all fit in round 1
        min_available_clients=NUM_CLIENTS,  # wait for all to connect
        initial_parameters=init_params,
        on_fit_config_fn=fit_config_factory(sel_rounds),
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1}
    )
    final_ndarrays = parameters_to_ndarrays(strategy.latest_parameters)
    acc=base.validation(final_ndarrays)
    print("Final ACC:", acc)
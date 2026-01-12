import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import torch
import flwr as fl
import random
import json
from pathlib import Path
from dataloader import CIFAR10BYOLClientData
from architectures import ResNet18Projv2
from SSL import SimpleBYOL
DATA_DIR = Path(__file__).resolve().parent / "data"   # absolute path
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()


class FedClient(fl.client.NumPyClient):
    def __init__(self, cid: int, num_partitions: int, local_epochs: int, batch_size: int):
        self.weight_storage="local_weights/"
        self.embedding_size=128
        self.cid = int(cid)
        self.target_path=self.weight_storage+"target"+str(self.cid)+".pt"
        self.EMA=ResNet18Projv2(self.embedding_size)
        self.model=ResNet18Projv2(self.embedding_size)
        self.local_epochs = local_epochs
        data_obj = CIFAR10BYOLClientData(
        num_clients=num_partitions,
        cid=self.cid,
        batch_size=128,
        keep_labels=False,
        data_dir="./data",
        seed=12345,
        device=DEVICE
        )
        self.global_ema_decay = 0.996
        self.SSL_trainer = SimpleBYOL(
            online_encoder=self.model,
            target_encoder=self.EMA,
            emb_dim=128,
            lr=3e-4,
            moving_average_decay=0.996,
            use_ema=True,        )
        self.train_loader, self.val_loader = data_obj.get_loaders()

    def fit(self, parameters, config):
        self.model.set_parameters(parameters)
        FFC=self.load_local()
        self._ema_target_with_global(parameters, decay=0.7)
        train_loss= self.SSL_trainer.train(self.train_loader, epochs=self.local_epochs)
        self.save_local()
        print(self.cid,train_loss)
        return self.model.get_parameters(), len(self.train_loader.dataset), {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        self.model.set_parameters(parameters)
        return 0.0, len(self.val_loader.dataset), {"accuracy": 0.0}
    

    def _ema_target_with_global(self, global_parameters, decay: float = None):
        """
        EMA_target = decay * EMA_target_old + (1 - decay) * global_model
        """
        if decay is None:
            decay = self.global_ema_decay

        device = next(self.EMA.parameters()).device
        global_tensors = [torch.from_numpy(p).to(device) for p in global_parameters]

        with torch.no_grad():
            for ema_param, global_param in zip(self.EMA.parameters(), global_tensors):
                ema_param.mul_(decay).add_(global_param, alpha=1.0 - decay)

    def get_weights(self):
        return self.model.get_parameters()
    
    def load_local(self):
        try:
            self.EMA.load_state_dict(torch.load(self.target_path, map_location=DEVICE))
            return True
        except:
            return False

    def save_local(self):
        torch.save(self.EMA.state_dict(), self.target_path)
    


if __name__=="__main__":
    clinet=FedClient(1,2,3,64)
    clinet.fit(None, None)

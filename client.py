import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import torch
import flwr as fl
from model import Net,Autoencoder, ML_functions, VQVAE
import random
import json
from pathlib import Path
from utils import vqvae_utils
DATA_DIR = Path(__file__).resolve().parent / "data"   # absolute path
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()


class FlowerClient(fl.client.NumPyClient, ML_functions):
    def __init__(self, cid: int, num_partitions: int, local_epochs: int, batch_size: int):
        self.embedding_size=512
        self.cid = int(cid)
        self.num_classes = 10
        self.n_classes=1
        self.noise_level = int(round(100 * (self.cid % num_partitions) / max(1, (num_partitions - 1))))
        rng = random.Random(self.cid)
        self.classes=rng.sample(range(self.num_classes), self.n_classes)
        self.classes_json=json.dumps([int(c) for c in self.classes])
        self.net = Net()
        self.trainloader, self.valloader = self.simple_load_data(self.cid, num_partitions, batch_size, self.noise_level, self.classes)
        self.local_epochs = local_epochs
        
        self.counts = torch.zeros(self.num_classes, dtype=torch.long)

    def test_local_data(self, num_samples: int, roundnum:int) -> float:
        # Use an absolute path to avoid CWD issues
        base_dir = Path(__file__).resolve().parent
        #model_path = "model"+str(roundnum)+".pth"
        model_path = "model"+".pth"
        ae_path = base_dir / model_path
        autoenc = VQVAE(dim=3, n_h=64, e_dim=32, n_embed=self.embedding_size,decay=0.99)
        state = torch.load(ae_path, map_location=DEVICE)   # ae_path = base_dir / "model.pth"
        autoenc.load_state_dict(state, strict=True)
        autoenc.eval()
        #return autoenc.recon_error_random_eval(self.trainloader, num_samples)
        hist= autoenc.code_histogram_random_eval(self.trainloader, num_samples, dp_sigma=0, clip_count=500.0)
        p_i = hist.cpu().numpy()                  # <<< convert here
        p_t = np.load(f"hist_class_{roundnum}.npy") # already numpy
        sim01 = 1.0 - vqvae_utils.js_divergence(p_i, p_t) / np.log(2.0)
        return float(sim01)

    def get_parameters(self, config):
        return self.get_weights(self.net)

    def fit(self, parameters, config):
        #if config.get("phase") == "init":
        phase = config.get("phase", "train")
        if phase != "train":
            mse = self.test_local_data(100, roundnum=phase)
            return self.get_weights(self.net), 100, {"quality": mse, "noise_level": float(self.noise_level), "classes": self.classes_json}
        self.set_weights(self.net, parameters)
        train_loss, self.counts = self.net.train_one_round(self.net, self.trainloader, epochs=self.local_epochs, tar_counter=self.counts)
        return self.get_weights(self.net), len(self.trainloader.dataset), {"train_loss": train_loss, "noise_level": float(self.noise_level)}

    def evaluate(self, parameters, config):
        self.set_weights(self.net, parameters)
        loss, acc = self.validate(self.net, self.valloader)
        return loss, len(self.valloader.dataset), {"accuracy": acc}
    



"""""
Data evenness metric
N = self.counts.sum()
p = self.counts / N
p = p[p > 0]
H = -(p * p.log()).sum()
J = H / torch.log(torch.tensor(len(self.counts), dtype=torch.float))

self.set_weights(self.net, parameters)
before=np.concatenate([p.ravel() for p in self.get_weights(self.net)])
train_loss, self.counts = self.train_one_round(self.net, self.trainloader, epochs=self.local_epochs, tar_counter=self.counts)
after=np.concatenate([p.ravel() for p in self.get_weights(self.net)])
euclidean = np.linalg.norm(after - before)
relative  = euclidean / np.linalg.norm(before)
return self.get_weights(self.net), len(self.trainloader.dataset), {"train_loss": train_loss, "shannon":1/train_loss}


"""""
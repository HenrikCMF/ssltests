import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import torch
import random
from collections import Counter
from model import Autoencoder, ML_functions, Net, VQVAE
from pathlib import Path
DATA_DIR = Path(__file__).resolve().parent / "data"   # absolute path
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()

class server(ML_functions):
    def __init__(self, number_of_clients, batch_size, local_epochs):
        self.embedding_size=512
        self.num_classes=10
        self.n_classes=3
        self.num_clients=number_of_clients
        self.batch_size=batch_size
        self.epochs=local_epochs
        #self.net = Autoencoder(self.num_classes, dim=3)
        self.net=VQVAE(dim=3, n_h=64, e_dim=32, n_embed=self.embedding_size,decay=0.99)
        rng = random.Random(1)
        self.classes=rng.sample(range(self.num_classes), self.n_classes)
        #self.classes=2
        print(self.classes)
        self.trainloader, self.valloader = self.simple_load_data(self.num_clients, self.num_clients+1, self.batch_size,0, self.classes, True)
        
    @torch.no_grad()
    def server_target_histogram(self,vqvae_model, ref_loader, n_embed: int, batches: int = 8):
        vqvae_model.eval()
        counts = torch.zeros(n_embed, device=DEVICE)
        seen = 0
        for b, (x, _) in enumerate(ref_loader):
            x = x.to(DEVICE, non_blocking=True)
            z = vqvae_model.encoder(x)
            _, _, _, codes = vqvae_model.quant(z)  # (B,H,W)
            binc = torch.bincount(codes.view(-1), minlength=n_embed)
            counts += binc
            seen += 1
            if b + 1 >= batches: break  # keep it tiny/constant
        hist = counts / (counts.sum() + 1e-8)
        return hist.detach().cpu().numpy()


    def get_parameters(self, config):
        return self.get_weights(self.net)
    
    def fit(self):
        roundnum=1
        self.trainloader, self.valloader = self.simple_load_data(self.num_clients, self.num_clients+1, self.batch_size,0, self.classes, True)
        n_samples = len(self.trainloader.dataset)
        train_loss= self.net.train_one_round(self.net, self.trainloader, self.epochs, use_ssim=False)
        torch.save(self.net.state_dict(), "model"+".pth")
        for i in self.classes:
            self.trainloader, self.valloader = self.simple_load_data(self.num_clients, self.num_clients+1, self.batch_size,0, i, True)
            hist=self.server_target_histogram(self.net, self.valloader, n_embed=self.embedding_size, batches=8)
            np.save("hist_class_"+str(roundnum)+".npy", hist)
            roundnum+=1
        self.trainloader, self.valloader = self.simple_load_data(self.num_clients, self.num_clients+1, self.batch_size,0, self.classes, True)

    def set_model_params_from_ndarrays(self,model: torch.nn.Module, ndarrays):
        # map ndarrays to the model's state_dict keys in order
        keys = list(model.state_dict().keys())
        state_dict = {k: torch.tensor(v) for k, v in zip(keys, ndarrays)}
        model.load_state_dict(state_dict, strict=True)

    def validation(self, parameters):
        def count_classes(dataloader, device="cpu"):
            counts = Counter()
            for _, y in dataloader:
                # y is a tensor of class indices
                y = y.to("cpu")   # move to CPU to be safe
                counts.update(y.tolist())
            return counts

        # usage
        class_counts = count_classes(self.valloader)
        print("Validation class distribution:")
        for cls, num in sorted(class_counts.items()):
            print(f"Class {cls}: {num}")
        c_model=Net()
        self.set_model_params_from_ndarrays(c_model, parameters)
        c_model.eval()
        c_model.to(DEVICE)
        correct = 0
        total = 0
        with torch.no_grad():  # disable autograd for speed/memory
            for x, y in self.valloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                outputs = c_model(x)
                # get predicted class indices
                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total
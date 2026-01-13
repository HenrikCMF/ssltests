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
        self.embedding_size=2048
        self.cid = int(cid)
        self.EMA_target_path=self.weight_storage+"target"+str(self.cid)+".pt"
        self.model_target_path=self.weight_storage+"model"+str(self.cid)+".pt"
        self.pred_path = self.weight_storage + "pred" + str(self.cid) + ".pt"
        self.EMA=ResNet18Projv2(self.embedding_size)
        self.model=ResNet18Projv2(self.embedding_size)
        self.local_epochs = local_epochs
        data_obj = CIFAR10BYOLClientData(
        num_clients=num_partitions,
        cid=self.cid,
        batch_size=batch_size,
        num_workers=5,
        keep_labels=False,
        data_dir="./data",
        seed=12345,
        device=DEVICE
        )
        self.global_ema_decay = 0.996
        self.train_loader, self.val_loader = data_obj.get_loaders()
        self.SSL_trainer = SimpleBYOL(
            online_encoder=self.model,
            target_encoder=self.EMA,
            emb_dim=self.embedding_size,
            lr=3e-4,
            moving_average_decay=0.996,
            use_ema=True,
            local_epochs=self.local_epochs,
            dataset_len=len(self.train_loader.dataset),
                            )
        

    def fit(self, parameters, config):
        # Server-provided FedEMA state
        selected_prev = bool(int(config.get("selected_prev", 0)))
        lambda_k = float(config.get("lambda_k", -1.0))
        lambda_k = None if lambda_k < 0 else lambda_k

        has_local = self.load_local()

        # Case A: first time OR not selected in r-1 OR lambda_k is null  -> reset to global
        if (not has_local) or (not selected_prev) or (lambda_k is None):
            self.set_federated_parameters(parameters)
            # init target = online (Wt_k <- Wg)
            self.EMA.load_state_dict(self.model.state_dict())

        # Case B: FedEMA divergence-aware dynamic mu update (Eq. 1-3 + Alg 1 line 16-18)
        else:
            # Snapshot previous local (r-1) BEFORE loading global
            local_model_sd = self._state_dict_clone(self.model.state_dict())
            local_pred_sd  = self._state_dict_clone(self.SSL_trainer.predictor.state_dict())

            # Load current global (r)
            self.set_federated_parameters(parameters)
            global_model_sd = self._state_dict_clone(self.model.state_dict())
            global_pred_sd  = self._state_dict_clone(self.SSL_trainer.predictor.state_dict())

            # divergence = ||Wg^r - Wk^{r-1}|| (encoder only)
            with torch.no_grad():
                s = 0.0
                for k, g in global_model_sd.items():
                    l = local_model_sd[k]
                    diff = (g.float() - l.float())
                    s += float((diff * diff).sum().item())
                div = (s ** 0.5)

            mu = min(lambda_k * div, 1.0)

            # Wk^r <- mu * Wk^{r-1} + (1-mu) * Wg^r  (encoder)
            self.model.load_state_dict(self._ema_blend_state(local_model_sd, global_model_sd, mu))

            # Wp_k^r <- mu * Wp_k^{r-1} + (1-mu) * Wp_g^r  (predictor)
            self.SSL_trainer.predictor.load_state_dict(self._ema_blend_state(local_pred_sd, global_pred_sd, mu))

            # IMPORTANT: do NOT overwrite self.EMA here (target stays local; updated per mini-batch in BYOL)

        train_loss = self.SSL_trainer.train(self.train_loader, epochs=self.local_epochs)
        self.save_local()
        return self.get_federated_parameters(), len(self.train_loader.dataset), {"train_loss": train_loss}


    def evaluate(self, parameters, config):
        self.set_federated_parameters(parameters)
        loss = self.SSL_trainer.evaluate(self.val_loader)  # you need to implement this
        return float(loss), len(self.val_loader.dataset), {"val_loss": float(loss)}

    

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

    
    def load_local(self):
        try:
            self.EMA.load_state_dict(torch.load(self.EMA_target_path, map_location=DEVICE))
            self.model.load_state_dict(torch.load(self.model_target_path, map_location=DEVICE))
            self.SSL_trainer.predictor.load_state_dict(torch.load(self.pred_path, map_location=DEVICE))
            return True
        except:
            return False

    def save_local(self):
        torch.save(self.EMA.state_dict(), self.EMA_target_path)
        torch.save(self.model.state_dict(), self.model_target_path)
        torch.save(self.SSL_trainer.predictor.state_dict(), self.pred_path)


    def set_federated_parameters(self, parameters):
        n_model = self._num_params(self.model)
        self._set_params(self.model, parameters[:n_model])
        self._set_params(self.SSL_trainer.predictor, parameters[n_model:])


    def get_federated_parameters(self):
        return self._get_params(self.model) + self._get_params(self.SSL_trainer.predictor)


    def _get_params(self,module: torch.nn.Module):
        # Ordered list of numpy arrays
        return [p.detach().cpu().numpy() for p in module.parameters()]

    def _set_params(self,module: torch.nn.Module, params_list):
        # Copy numpy arrays into module parameters
        with torch.no_grad():
            for p, w in zip(module.parameters(), params_list):
                p.copy_(torch.from_numpy(w).to(p.device))

    def _num_params(self,module: torch.nn.Module) -> int:
        return sum(1 for _ in module.parameters())
    
    def _state_dict_clone(self,sd):
        return {k: v.detach().clone() for k, v in sd.items()}

    def _ema_blend_state(self,local_sd, global_sd, mu: float):
        return {k: mu * local_sd[k] + (1 - mu) * global_sd[k] for k in global_sd.keys()}
    


if __name__=="__main__":
    #clinet=FedClient(1,2,3,64)
    #clinet.fit(None, None)
    pass

"""""
def fit(self, parameters, config):
        FFC = self.load_local()
        if FFC:
            # local copies
            local_model_sd = self._state_dict_clone(self.model.state_dict())
            local_pred_sd  = self._state_dict_clone(self.SSL_trainer.predictor.state_dict())

            # load global into both
            self.set_federated_parameters(parameters)
            global_model_sd = self.model.state_dict()
            global_pred_sd  = self.SSL_trainer.predictor.state_dict()

            mu = 0.7
            self.model.load_state_dict(self._ema_blend_state(local_model_sd, global_model_sd, mu))
            self.SSL_trainer.predictor.load_state_dict(self._ema_blend_state(local_pred_sd, global_pred_sd, mu))

            # do NOT touch self.EMA here

        else:
            self.set_federated_parameters(parameters)

            # initialize target = online
            self.EMA.load_state_dict(self.model.state_dict())
        train_loss= self.SSL_trainer.train(self.train_loader, epochs=self.local_epochs)
        #self.SSL_trainer.
        self.save_local()
        #print(self.cid,train_loss)
        return self.get_federated_parameters(), len(self.train_loader.dataset), {"train_loss": train_loss}
"""
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import torch
import flwr as fl
from pathlib import Path
from dataloader import CIFAR10BYOLClientData
from architectures import build_models
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
    def __init__(self, cid: int, num_partitions: int, local_epochs: int, batch_size: int, total_rounds: int):
        self.weight_storage="local_weights/"
        self.embedding_size=2048#2048
        self.total_rounds=total_rounds
        self.cid = int(cid)
        self.EMA_target_path=self.weight_storage+"target"+str(self.cid)+".pt"
        self.model_target_path=self.weight_storage+"model"+str(self.cid)+".pt"
        self.pred_path = self.weight_storage + "pred" + str(self.cid) + ".pt"
        #self.EMA=ResNet18Projv2(self.embedding_size)
        #self.model=ResNet18Projv2(self.embedding_size)
        self.local_epochs = local_epochs
        data_obj = CIFAR10BYOLClientData(
        num_clients=num_partitions,
        cid=self.cid,
        batch_size=batch_size,
        num_workers=8,
        keep_labels=False,
        data_dir="./data",
        seed=12345,
        device=DEVICE
        )
        self.global_ema_decay = 0.99
        self.train_loader, self.val_loader = data_obj.get_loaders()
        self.model, self.EMA, self.predictor = build_models(emb_dim=self.embedding_size)
        
        

    def fit(self, parameters, config):
        # Server-provided FedEMA state
        selected_prev = bool(int(config.get("selected_prev", 0)))
        lambda_k = float(config.get("lambda_k", -1.0)) 
        lambda_k = None if lambda_k < 0 else lambda_k
        #lambda_k=0.8
        NUM_ROUND=float(config.get("server_round", -1.0))
        has_local = self.load_local()
        
        # Case A: first time OR not selected in r-1 OR lambda_k is null  -> reset to global
        if (not has_local) or (not selected_prev) or (lambda_k is None):
            #print(self.cid, "A")
            self.set_federated_parameters(parameters)
            # init target = online (Wt_k <- Wg)
            self.EMA.load_state_dict(self.model.state_dict())

        # Case B: FedEMA divergence-aware dynamic mu update (Eq. 1-3 + Alg 1 line 16-18)
        else:
            
            # Snapshot previous local (r-1) BEFORE loading global
            local_model_sd = self._state_dict_clone(self.model.state_dict())
            local_pred_sd  = self._state_dict_clone(self.predictor.state_dict())

            # Load current global (r)
            self.set_federated_parameters(parameters)
            global_model_sd = self._state_dict_clone(self.model.state_dict())
            global_pred_sd  = self._state_dict_clone(self.predictor.state_dict())

            # divergence = ||Wg^r - Wk^{r-1}|| (encoder only)
            with torch.no_grad():
                s = 0.0
                for k, g in global_model_sd.items():
                    l = local_model_sd[k]
                    if torch.is_tensor(g) and g.is_floating_point():
                        diff = (g.float() - l.float())
                        s += float((diff * diff).sum().item())
                div = (s ** 0.5)

                mu = max(min(lambda_k * div, 1.0),0.2)
                #mu=0.6
            #print(f"Client {self.cid} round {NUM_ROUND} | lambda_k = {lambda_k} | selected_prev = {selected_prev} | has_local = {has_local}")
            # Wk^r <- mu * Wk^{r-1} + (1-mu) * Wg^r  (encoder)
            self.model.load_state_dict(self._ema_blend_state(local_model_sd, global_model_sd, mu))
            

            # Wp_k^r <- mu * Wp_k^{r-1} + (1-mu) * Wp_g^r  (predictor)
            self.predictor.load_state_dict(self._ema_blend_state(local_pred_sd, global_pred_sd, mu))

            #self.EMA.load_state_dict(self.model.state_dict())

        self.SSL_trainer = SimpleBYOL(
            online_encoder=self.model,
            target_encoder=self.EMA,
            predictor=self.predictor,
            emb_dim=self.embedding_size,
            lr=0.032,#3e-4,
            moving_average_decay=0.99,
            use_ema=True,
            local_epochs=self.local_epochs,
            dataset_len=len(self.train_loader),
                            )
        train_loss = self.SSL_trainer.train(self.train_loader, epochs=self.local_epochs, server_round=NUM_ROUND)
        self.save_local()
        return self.get_federated_parameters(), len(self.train_loader.dataset), {"train_loss": train_loss}


    def evaluate(self, parameters, config):
        self.set_federated_parameters(parameters)
        #loss = self.SSL_trainer.evaluate(self.val_loader)
        loss=1
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
            self.predictor.load_state_dict(torch.load(self.pred_path, map_location=DEVICE))
            return True
        except:
            return False

    def save_local(self):
        torch.save(self.EMA.state_dict(), self.EMA_target_path)
        torch.save(self.model.state_dict(), self.model_target_path)
        torch.save(self.predictor.state_dict(), self.pred_path)

    def set_federated_parameters(self, parameters):
        n_model = self._num_sd_floats(self.model)
        self._set_sd_floats(self.model, parameters[:n_model])
        self._set_sd_floats(self.predictor, parameters[n_model:])

    def get_federated_parameters(self):
        return self._get_sd_floats(self.model) + self._get_sd_floats(self.predictor)

    def _set_params(self,module: torch.nn.Module, params_list):
        # Copy numpy arrays into module parameters
        with torch.no_grad():
            for p, w in zip(module.parameters(), params_list):
                p.copy_(torch.from_numpy(w).to(p.device))

    def _num_params(self,module: torch.nn.Module) -> int:
        return sum(1 for _ in module.parameters())
    
    def _state_dict_clone(self,sd):
        return {k: v.detach().clone() for k, v in sd.items()}

    def _ema_blend_state(self, local_sd, global_sd, mu: float):
        """
        Blend ONLY float tensors. For non-float entries (ints), keep global.
        """
        out = {}
        for k in global_sd.keys():
            g = global_sd[k]
            l = local_sd[k]
            if torch.is_tensor(g) and g.is_floating_point():
                out[k] = mu * l + (1.0 - mu) * g
            else:
                out[k] = g
        return out
    

    def _state_dict_keys_float(self, module: torch.nn.Module):
        """Stable key list of float tensors in state_dict (params + float buffers like BN running stats)."""
        sd = module.state_dict()
        keys = []
        for k, v in sd.items():
            if torch.is_tensor(v) and v.is_floating_point():
                keys.append(k)
        return keys

    def _get_sd_floats(self, module: torch.nn.Module):
        """Return list[np.ndarray] in a stable order corresponding to float state_dict entries."""
        sd = module.state_dict()
        keys = self._state_dict_keys_float(module)
        return [sd[k].detach().cpu().numpy() for k in keys]

    def _set_sd_floats(self, module: torch.nn.Module, arrays):
        """Load list[np.ndarray] into the module's float state_dict entries (same stable order)."""
        sd = module.state_dict()
        keys = self._state_dict_keys_float(module)

        if len(arrays) != len(keys):
            raise ValueError(
                f"Payload mismatch for {module.__class__.__name__}: "
                f"got {len(arrays)} arrays, expected {len(keys)}"
            )

        new_sd = {}
        for k, arr in zip(keys, arrays):
            t = torch.from_numpy(arr).to(sd[k].device)
            # keep dtype/shape exactly as expected
            new_sd[k] = t.to(dtype=sd[k].dtype).view_as(sd[k])

        # strict=False so non-float buffers (e.g., num_batches_tracked) remain intact
        module.load_state_dict(new_sd, strict=False)

    def _num_sd_floats(self, module: torch.nn.Module) -> int:
        return len(self._state_dict_keys_float(module))
    


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
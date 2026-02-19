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
import pickle
DATA_DIR = Path(__file__).resolve().parent / "data"   # absolute path
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()

class FedClient(fl.client.NumPyClient):
    def __init__(self, cid: int, num_partitions: int, local_epochs: int, batch_size: int, total_rounds: int, embedding_size:int):
        self.weight_storage="local_weights/"
        self.embedding_size=embedding_size
        self.total_rounds=total_rounds
        self.cid = int(cid)
        self.EMA_target_path=self.weight_storage+"target"+str(self.cid)+".pt"
        self.model_target_path=self.weight_storage+"model"+str(self.cid)+".pt"
        self.pred_path = self.weight_storage + "pred" + str(self.cid) + ".pt"
        self.p_model_path = self.weight_storage + "p_model"+ str(self.cid) +".pt"
        self.p_pred_path = self.weight_storage + "p_pred"+ str(self.cid) +".pt"
        self.optim_path = self.weight_storage + "optimizer"+ str(self.cid) +".pt"
        self.lambda_path = self.weight_storage + "lambda"+ str(self.cid) +".pkl"
        #self.EMA=ResNet18Projv2(self.embedding_size)
        #self.model=ResNet18Projv2(self.embedding_size)
        self.local_epochs = local_epochs
        data_obj = CIFAR10BYOLClientData(
        num_clients=num_partitions,
        cid=self.cid,
        classes_per_client=2,
        batch_size=batch_size,
        num_workers=8,
        keep_labels=False,
        data_dir="./data",
        seed=12345,
        device=DEVICE
        )
        self.global_ema_decay = 0.99
        self.learning_rate=0.032
        self.tau=0.7
        self.train_loader, self.val_loader = data_obj.get_loaders()
        self.model, self.EMA, self.predictor, self.p_model, self.p_pred = build_models(emb_dim=self.embedding_size)

        
        

    def fit(self, parameters, config):
        # Server-provided FedEMA state
        selected_prev = bool(int(config.get("selected_prev", 0)))
        #lambda_k = float(config.get("lambda_k", -1.0)) 
        #lambda_k = None if lambda_k < 0 else lambda_k
        #lambda_k=0.8
        self.NUM_ROUND=float(config.get("server_round", -1.0))
        has_local = self.load_local()
        # Case A: first time OR not selected in r-1 OR lambda_k is null  -> reset to global
        if (not has_local) or (not selected_prev):
            #print(self.cid, "A")
            self.set_federated_parameters(parameters)
            # init target = online (Wt_k <- Wg)
            self.EMA.load_state_dict(self.model.state_dict())
            self.p_model.load_state_dict(self.model.state_dict())
            self.p_pred.load_state_dict(self.predictor.state_dict())
        # Case B: FedEMA divergence-aware dynamic mu update (Eq. 1-3 + Alg 1 line 16-18)
        else:
            
            # Snapshot previous local (r-1) BEFORE loading global
            local_model_sd = self._state_dict_clone(self.model.state_dict())
            local_pred_sd  = self._state_dict_clone(self.predictor.state_dict())
            p_local_model_sd=self._state_dict_clone(self.p_model.state_dict())
            p_pred_sd = self._state_dict_clone(self.p_pred.state_dict())
           # Load current global (r)

            #p_combined_sd = {f"model.{k}": v for k, v in p_local_model_sd.items()} | {f"pred.{k}": v for k, v in p_pred_sd.items()}
            #l_combined_sd = {f"model.{k}": v for k, v in local_model_sd.items()} | {f"pred.{k}": v for k, v in local_pred_sd.items()}
            #div = self.model_l2_relative_distance(p_combined_sd,l_combined_sd)

            self.set_federated_parameters(parameters)
            global_model_sd = self._state_dict_clone(self.model.state_dict())
            global_pred_sd  = self._state_dict_clone(self.predictor.state_dict())
            
            p_combined_sd = {f"model.{k}": v for k, v in global_model_sd.items()} | {f"pred.{k}": v for k, v in global_pred_sd.items()}
            l_combined_sd = {f"model.{k}": v for k, v in local_model_sd.items()} | {f"pred.{k}": v for k, v in local_pred_sd.items()}
            div = self.model_l2_relative_distance(p_combined_sd,l_combined_sd)
            if self.lambda_k is None:
                self.lambda_k=(self.tau)/div
                #self.lambda_k=0.8
                with open(self.lambda_path, 'wb') as file:
                    pickle.dump(self.lambda_k, file)
            mu =  1#min(self.lambda_k * div, 1)
            if self.cid==0:
                print(div,mu,self.lambda_k)

            self.model.load_state_dict(self._ema_blend_state(local_model_sd, global_model_sd, mu))
            self.p_model.load_state_dict(self.model.state_dict())
            

            # Wp_k^r <- mu * Wp_k^{r-1} + (1-mu) * Wp_g^r  (predictor)
            self.predictor.load_state_dict(self._ema_blend_state(local_pred_sd, global_pred_sd, mu, predictor=True))
            self.p_pred.load_state_dict(self.predictor.state_dict())

        self.SSL_trainer = SimpleBYOL(
            online_encoder=self.model,
            target_encoder=self.EMA,
            predictor=self.predictor,
            lr=self.learning_rate,#3e-4,
            moving_average_decay=self.global_ema_decay,
            use_ema=True,
            local_epochs=self.local_epochs,
            dataset_len=len(self.train_loader),
            total_rounds=self.total_rounds,
                            )
        if has_local:
            self.SSL_trainer.optimizer.load_state_dict(torch.load(self.optim_path))
        train_loss = self.SSL_trainer.train(self.train_loader, epochs=self.local_epochs, server_round=self.NUM_ROUND)
        self.save_local()
        return self.get_federated_parameters(), len(self.train_loader.dataset), {"train_loss": train_loss}


    def evaluate(self, parameters, config):
        self.set_federated_parameters(parameters)
        #loss = self.SSL_trainer.evaluate(self.val_loader)
        loss=1
        return float(loss), len(self.val_loader.dataset), {"val_loss": float(loss)}


    
    def load_local(self):
        try:
            self.EMA.load_state_dict(torch.load(self.EMA_target_path, map_location=DEVICE))
            self.model.load_state_dict(torch.load(self.model_target_path, map_location=DEVICE))
            self.predictor.load_state_dict(torch.load(self.pred_path, map_location=DEVICE))
            self.p_model.load_state_dict(torch.load(self.p_model_path, map_location=DEVICE))
            self.p_pred.load_state_dict(torch.load(self.p_pred_path, map_location=DEVICE))
            if self.NUM_ROUND>2:
                with open(self.lambda_path, 'rb') as file:
                    self.lambda_k = pickle.load(file)
            else:
                self.lambda_k= None
            return True
        except:
            self.lambda_k=None
            return False

    def save_local(self):
        torch.save(self.EMA.state_dict(), self.EMA_target_path)
        torch.save(self.model.state_dict(), self.model_target_path)
        torch.save(self.predictor.state_dict(), self.pred_path)
        torch.save(self.p_model.state_dict(), self.p_model_path)
        torch.save(self.p_pred.state_dict(), self.p_pred_path)
        torch.save(self.SSL_trainer.optimizer.state_dict(), self.optim_path)

    def set_federated_parameters(self, parameters):
        n_model = self._num_sd_floats(self.model)
        self._set_sd_floats(self.model, parameters[:n_model])
        self._set_sd_floats(self.predictor, parameters[n_model:])
        #model_param_count = sum(1 for _ in self.model.parameters())+sum(1 for _ in self.predictor.parameters())
        #model_arrs   = parameters[:model_param_count]
        #predict_arrs = parameters[model_param_count:]
        #for p, arr in zip(self.model.parameters(), model_arrs):
        #    p.data.copy_(torch.from_numpy(arr).to(p.device))
        #for p, arr in zip(self.predictor.parameters(), predict_arrs):
        #    p.data.copy_(torch.from_numpy(arr).to(p.device))

    def get_federated_parameters(self):
        return self._get_sd_floats(self.model) + self._get_sd_floats(self.predictor)
        #model_params   = [p.detach().cpu().numpy() for p in self.model.parameters()]
        #predict_params = [p.detach().cpu().numpy() for p in self.predictor.parameters()]
        #return model_params + predict_params

    
    def _state_dict_clone(self,sd):
        return {k: v.detach().clone() for k, v in sd.items()}


    def _ema_blend_state(self, local_sd, global_sd, mu: float, predictor=False):
        out = {}
        for k in global_sd.keys():
            g = global_sd[k]
            l = local_sd[k]
            if torch.is_tensor(g) and g.is_floating_point():
                #out[k] = mu * l + (1.0 - mu) * g
                out[k] = (1.0 - mu) * l + mu * g
            elif 'num_batches_tracked' in k:
                # Keep local num_batches_tracked, not global
                out[k] = l
            else:
                out[k] = g
        return out


    def _state_dict_keys_float(self, module: torch.nn.Module):
        """Stable key list of float tensors in state_dict (params + float buffers like BN running stats)."""
        sd = module.state_dict()
        keys = []
        for k, v in sd.items():
            #if "running_mean" in k or "running_var" in k: 
            #    continue
            if torch.is_tensor(v) and v.is_floating_point():
                keys.append(k)
        return keys
    

    def _state_dict_keys_float_bn(self,module: torch.nn.Module):
        sd = module.state_dict()
        keys = []
        for k, v in sd.items():
            if not (torch.is_tensor(v) and v.is_floating_point()):
                continue
            # Exclude BN running stats buffers
            if ("running_mean" in k) or ("running_var" in k):
                continue
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
    

    def model_l2_relative_distance(self, global_sd: dict, local_sd: dict) -> float:
        with torch.no_grad():
            #total_sq = 0.0
            #count = 0
            size = 0
            total_distance = 0
            for name, param in local_sd.items():
            #for name, local_t in local_sd.items():
                if 'conv' in name and 'weight' in name:
                    total_distance+=torch.dist(local_sd[name].detach().clone().view(1, -1),
                                                global_sd[name].detach().clone().view(1, -1), 2)
                    #global_t = global_sd[name]
                    #diff = (local_t.detach().view(-1).double()- global_t.detach().view(-1).double())
                    #total_sq += float(torch.dot(diff, diff))
                    #count += diff.numel()
                    size += 1
            distance = total_distance / size
            return distance
            #return float(total_sq ** 0.5)


    def model_l2_relative_distance_2(self, global_sd: dict, local_sd: dict) -> float:
        """
        Paper-aligned divergence for FedEMA:
        Compute ||W_g - W_k||_2 over the *encoder parameters*.

        Assumptions:
        - If your dict contains both encoder and predictor entries, encoder keys
            are prefixed with 'model.' (as in your combined_sd usage).
        - Excludes BN running stats buffers and num_batches_tracked.
        - Includes weights *and biases* (all floating-point tensors).

        Returns:
        float L2 norm
        """
        with torch.no_grad():
            total_sq = 0.0
            used = 0

            for name, local_t in local_sd.items():
                # Keep ONLY encoder entries if you pass combined dicts
                if not name.startswith("model."):
                    continue

                # Must exist in global as well
                if name not in global_sd:
                    continue

                # Exclude BN running stats / counters
                if ("running_mean" in name) or ("running_var" in name) or ("num_batches_tracked" in name):
                    continue

                global_t = global_sd[name]

                # Only compute over floating point tensors
                if not (torch.is_tensor(local_t) and torch.is_tensor(global_t)):
                    continue
                if not (local_t.is_floating_point() and global_t.is_floating_point()):
                    continue

                # Accumulate squared L2 over all elements
                diff = (local_t.detach().reshape(-1).double() - global_t.detach().reshape(-1).double())
                total_sq += float(torch.dot(diff, diff))
                used += diff.numel()

            if used == 0:
                return 0.0

            return float(total_sq ** 0.5)
    
def _state_dict_keys_float(module: torch.nn.Module):
        sd = module.state_dict()
        keys = []
        for k, v in sd.items():
            if not (torch.is_tensor(v) and v.is_floating_point()):
                continue
            # Exclude BN running stats buffers
            if ("running_mean" in k) or ("running_var" in k):
                continue
            keys.append(k)
        return keys

if __name__=="__main__":
    #clinet=FedClient(1,2,3,64)
    #clinet.fit(None, None)
    pass


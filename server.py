# fedema_strategy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import flwr as fl
from flwr.common import FitIns, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms as T
import flwr as fl
from flwr.common import parameters_to_ndarrays
from dataloader import build_eval_loaders
#from dataloader import build_server_eval_loaders
import numpy as np
from typing import List

def l2_norm_between(a: List[np.ndarray], b: List[np.ndarray]) -> float:
    # sqrt(sum ||ai - bi||^2)
    s = 0.0
    for ai, bi in zip(a, b):
        diff = ai.astype(np.float64) - bi.astype(np.float64)
        s += float(np.sum(diff * diff))
    return float(np.sqrt(s))

def l2_norm_between_filtered(
    a: List[np.ndarray], b: List[np.ndarray], include_indices: List[int]
) -> float:
    """L2 norm but only over a subset of parameter indices (e.g. skip BN stats)."""
    s = 0.0
    #for i in include_indices:
    #    ai = a[i]
    #    bi = b[i]
    #    diff = ai.astype(np.float64) - bi.astype(np.float64)
    #    s += float(np.sum(diff * diff))
    #return float(np.sqrt(s))
    size = 0
    total_distance = 0
    num_layers = len(include_indices)
    for i in include_indices:
        a_tensor = torch.from_numpy(a[i])
        b_tensor = torch.from_numpy(b[i])
        total_distance += torch.dist(a_tensor, b_tensor)
        size+=1
    return total_distance / size
        #ai = a[i].astype(np.float64)
        #bi = b[i].astype(np.float64)
        #diff = ai - bi
        #s += np.sum(diff * diff)  # sum sq
    #return float(np.sqrt(s / num_layers)) if num_layers > 0 else 0.0

@torch.no_grad()
def calibrate_bn(model, loader, device, num_batches=50):
    model = model.to(device)
    model.train()
    # Only need to forward; no gradients
    for i, (x, _) in enumerate(loader):
        if i >= num_batches:
            break
        x = x.to(device, non_blocking=True)
        # Forward through backbone to update BN stats
        _ = model.encode_backbone(x, normalize=False)
    model.eval()

@torch.no_grad()
def extract_feats(encoder, loader, device, normalize=True, use_pred=False):
    encoder = encoder.to(device)
    encoder.eval()
    for module in encoder.modules():
        if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
            module.track_running_stats = True
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)

        # Paper eval: remove encoder MLP -> use backbone representation
        if use_pred:
            z = encoder(x, normalize=False)
            if normalize:
                z = F.normalize(z, dim=1)
        else:
            z = encoder.encode_backbone(x, normalize=normalize)  # [B,512]


        feats.append(z.detach().cpu())
        labels.append(y)
    return torch.cat(feats, 0), torch.cat(labels, 0)

@torch.no_grad()
def knn_acc(train_feats, train_labels, test_feats, test_labels, k=200, temperature=0.1):
    # cosine sim because features are L2-normalized
    k = min(k, train_feats.shape[0])
    sims = test_feats @ train_feats.T
    topk_sims, topk_idx = sims.topk(k=k, dim=1)
    topk_labels = train_labels[topk_idx]

    weights = torch.exp(topk_sims / temperature)
    #weights = torch.ones_like(topk_sims)
    num_classes = int(train_labels.max().item()) + 1

    votes = torch.zeros(test_feats.size(0), num_classes, device=test_feats.device)
    for c in range(num_classes):
        votes[:, c] = (weights * (topk_labels == c).float()).sum(dim=1)

    pred = votes.argmax(dim=1)
    return (pred == test_labels).float().mean().item()

@dataclass
class FedEMAState:
    tau: float = 0.7          # autoscaler target expected mu
    eps: float = 1e-12
    lambda_k: Dict[str, Optional[float]] = None
    last_selected_round: Dict[str, int] = None

    def __post_init__(self):
        if self.lambda_k is None:
            self.lambda_k = {}
        if self.last_selected_round is None:
            self.last_selected_round = {}

class FedEMAStrategy(fl.server.strategy.FedAvg):
    """
    FedEMA (Algorithm 1):
      - Maintains per-client lambda_k (autoscaler)
      - Tells clients whether they were selected in r-1
      - Sends lambda_k to each selected client
      - Computes lambda_k once at earliest participation:
            lambda_k = tau / || W_g^{r+1} - W_k^r ||
        where W_k^r is the client model returned this round (encoder only).
    """

    def __init__(self, *, tau: float = 0.7, n_model_params: int, **kwargs):
        super().__init__(**kwargs)
        self.state = FedEMAState(tau=tau)
        self.n_model_params = n_model_params  # how many arrays belong to encoder model (not predictor)

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, FitIns]]:
        pairs = super().configure_fit(server_round, parameters, client_manager)

        configured: List[Tuple[fl.server.client_proxy.ClientProxy, FitIns]] = []
        for client, fitins in pairs:
            cid = client.cid

            # Were they selected in r-1?
            prev = self.state.last_selected_round.get(cid, None)
            selected_prev = (prev == server_round - 1)

            lam = self.state.lambda_k.get(cid, None)  # None => "null" in paper

            # Merge into config
            cfg = dict(fitins.config) if fitins.config is not None else {}
            cfg.update(
                {
                    "server_round": server_round,
                    "selected_prev": int(selected_prev),   # 1/0 for easy JSON
                    "lambda_k": (-1.0 if lam is None else float(lam)),
                    
                }
            )

            configured.append((client, FitIns(fitins.parameters, cfg)))

            # Mark selected this round (needed for next round)
            self.state.last_selected_round[cid] = server_round

        return configured

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is None:
            return None

        params_agg, metrics_agg = aggregated
        if params_agg is None:
            # No successful client updates this round -> skip
            return None
        global_nd = parameters_to_ndarrays(params_agg)
        global_model = global_nd[: self.n_model_params]

        # Autoscaler: set lambda_k once at earliest participation
        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            if self.state.lambda_k.get(cid, None) is not None:
                continue  # already set once

            client_nd = parameters_to_ndarrays(fit_res.parameters)
            client_model = client_nd[: self.n_model_params]

            div = l2_norm_between(global_model, client_model)
            #div = l2_norm_between_filtered(global_model, client_model, self.enc_div_indices)
            lam = self.state.tau / (div + self.state.eps)
            self.state.lambda_k[cid] = float(lam)

        return params_agg, metrics_agg

class FedEMAStrategyWithKnn(FedEMAStrategy):
    def __init__(self, data_dir="./data", k=200, temperature=0.1, eval_model=None, **kwargs):
        super().__init__(**kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.k = k
        self.temperature = temperature

        # Same encoder architecture as the clients’ encoder
        self.eval_model = eval_model

        self.train_ld, self.test_ld = build_eval_loaders(
            data_dir=data_dir, batch_size=512, num_workers=4
        )
        #self.label_ld, self.eval_ld = build_server_eval_loaders(
        #    data_dir=data_dir,
        #    batch_size=512,
        #    num_workers=4,
        #    num_clients=6,   # or pass explicitly via your config
        #    server_cid=5,
        #    classes_per_client=10,
            #seed=kwargs.get("seed", 12345),
        #    non_iid=False,  # recommended False for 10/class
        #    labeled_per_class=10,
        #    eval_on_remaining_train_plus_test=True,       # "whole rest of CIFAR-10"
        #)
        sd = self.eval_model.state_dict()
        self.enc_div_indices: List[int] = []
        idx = 0
        #for k, v in sd.items():
        #    if torch.is_tensor(v) and v.is_floating_point():
                #if 'running_mean' not in k and 'running_var' not in k:
        #        if 'conv' in k and 'weight' in k:
        #            self.enc_div_indices.append(idx)
        #        idx += 1

    def _enc_float_keys(self):
        sd = self.eval_model.state_dict()
        return [k for k, v in sd.items() if torch.is_tensor(v) and v.is_floating_point()]
        
    def _enc_float_keys_bn(self):
        sd = self.eval_model.state_dict()
        keys = []
        for k, v in sd.items():
            if not (torch.is_tensor(v) and v.is_floating_point()):
                continue
            # Exclude BN running stats buffers so the ndarray ordering matches the clients
            if ("running_mean" in k) or ("running_var" in k):
                continue
            keys.append(k)
        return keys

    def _load_encoder_from_ndarrays(self, nds):
        """
        nds contains [encoder floats..., predictor floats...]
        We only load the first self.n_model_params arrays into eval_model (float state_dict entries).
        """
        enc_nds = nds[: self.n_model_params]
        keys = self._enc_float_keys()
        if len(enc_nds) != len(keys):
            raise ValueError(f"Encoder payload mismatch: got {len(enc_nds)} arrays, expected {len(keys)}")

        sd = self.eval_model.state_dict()
        new_sd = {}
        for k, arr in zip(keys, enc_nds):
            t = torch.from_numpy(arr).to(sd[k].device)
            new_sd[k] = t.to(dtype=sd[k].dtype).view_as(sd[k])

        self.eval_model.load_state_dict(new_sd, strict=False)

    
        


    def _train_linear_probe(
        self,
        train_feats: torch.Tensor,
        train_labels: torch.Tensor,
        test_feats: torch.Tensor,
        test_labels: torch.Tensor,
        num_epochs: int = 10,      # ← increase to 50-100 for final eval
        lr: float = 0.5,
        batch_size: int = 1024,
        weight_decay: float = 5e-4,
    ) -> float:
        """Quick linear probing on frozen backbone features (512-dim)."""
        feat_dim = train_feats.shape[1]
        num_classes = int(train_labels.max().item()) + 1   # 10 for CIFAR-10

        classifier = nn.Linear(feat_dim, num_classes).to(self.device)
        
        optimizer = torch.optim.SGD(
            classifier.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
        )
        criterion = nn.CrossEntropyLoss().to(self.device)

        # DataLoaders (features are already on CPU from extract_feats)
        train_ds = TensorDataset(train_feats.to(self.device), train_labels.to(self.device))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

        classifier.train()
        for _ in range(num_epochs):
            for feats, labels in train_loader:
                optimizer.zero_grad()
                outputs = classifier(feats)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Evaluate
        classifier.eval()
        with torch.no_grad():
            test_outputs = classifier(test_feats.to(self.device))
            preds = test_outputs.argmax(dim=1)
            acc = (preds == test_labels.to(self.device)).float().mean().item()

        return acc

    def evaluate(self, server_round, parameters):
        if parameters is None:
            return None

        nds = parameters_to_ndarrays(parameters)
        if len(nds) == 0:
            return None

        self._load_encoder_from_ndarrays(nds)
        #calibrate_bn(self.eval_model, self.train_ld, self.device, num_batches=50)
        # Extract features once (backbone 512-dim, same as kNN)
        train_feats, train_labels = extract_feats(self.eval_model, self.train_ld, self.device, normalize=True, use_pred=False)
        test_feats,  test_labels  = extract_feats(self.eval_model, self.test_ld,  self.device, normalize=True, use_pred=False)
        #train_feats, train_labels = extract_feats(
        #    self.eval_model, self.label_ld, self.device, normalize=True, use_pred=False
        #)
        #test_feats, test_labels = extract_feats(
        #    self.eval_model, self.eval_ld, self.device, normalize=True, use_pred=False
        #)
        #p_train_feats, p_train_labels = extract_feats(self.eval_model, self.train_ld, self.device, normalize=True, use_pred=True)
        #p_test_feats,  p_test_labels  = extract_feats(self.eval_model, self.test_ld,  self.device, normalize=True, use_pred=True)


        # kNN (existing)
        #k_eff = min(self.k, train_feats.shape[0])
        b_acc_knn = knn_acc(
            train_feats, train_labels,
            test_feats,  test_labels,
            k=self.k, temperature=self.temperature
        )
        #pred_acc_knn = knn_acc(
        #    p_train_feats, p_train_labels,
        #    p_test_feats,  p_test_labels,
        #    k=self.k, temperature=self.temperature
        #)

        # ← NEW: Linear probing
        #acc_linear = self._train_linear_probe(
        #    train_feats, train_labels,
        #    test_feats,  test_labels,
        #    num_epochs=10,      # 5-10 = fast monitoring, 50-100 = more accurate
        #    lr=0.5,
        #)

        torch.save(self.eval_model.state_dict(), f"eval_model.pth")

        loss = 1.0 - b_acc_knn  # Flower still expects a loss; you can change to 1-acc_linear if you prefer

        #print(f"Round {server_round:3d} | kNN: {acc_knn*100:5.2f}% | Linear: {acc_linear*100:5.2f}%")

        return loss, {"knn_acc": b_acc_knn}#, "pred_knn_acc": pred_acc_knn}#, "linear_acc": acc_linear}
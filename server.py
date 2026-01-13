# fedema_strategy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import flwr as fl
from flwr.common import FitIns, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T
from architectures import ResNet18Projv2
import flwr as fl
from flwr.common import parameters_to_ndarrays
from dataloader import build_eval_loaders
import numpy as np
from typing import List

def l2_norm_between(a: List[np.ndarray], b: List[np.ndarray]) -> float:
    # sqrt(sum ||ai - bi||^2)
    s = 0.0
    for ai, bi in zip(a, b):
        diff = ai.astype(np.float64) - bi.astype(np.float64)
        s += float(np.sum(diff * diff))
    return float(np.sqrt(s))

@torch.no_grad()
def extract_feats(encoder, loader, device):
    encoder.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device)
        z = encoder.encode_backbone(x, normalize=True)  # [B,512], normalized
        feats.append(z.cpu())
        labels.append(y)
    return torch.cat(feats, 0), torch.cat(labels, 0)

@torch.no_grad()
def knn_acc(train_feats, train_labels, test_feats, test_labels, k=200, temperature=0.1):
    # cosine sim because features are L2-normalized
    sims = test_feats @ train_feats.T                    # [Ntest, Ntrain]
    topk_sims, topk_idx = sims.topk(k=k, dim=1)
    topk_labels = train_labels[topk_idx]

    weights = torch.exp(topk_sims / temperature)
    num_classes = int(train_labels.max().item()) + 1

    votes = torch.zeros(test_feats.size(0), num_classes)
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
        global_nd = parameters_to_ndarrays(params_agg)
        global_model = global_nd[: self.n_model_params]  # encoder arrays only

        # Autoscaler: set lambda_k once at earliest participation
        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            if self.state.lambda_k.get(cid, None) is not None:
                continue  # already set once

            client_nd = parameters_to_ndarrays(fit_res.parameters)
            client_model = client_nd[: self.n_model_params]

            div = l2_norm_between(global_model, client_model)
            lam = self.state.tau / (div + self.state.eps)
            self.state.lambda_k[cid] = float(lam)

        return params_agg, metrics_agg

class FedEMAStrategyWithKnn(FedEMAStrategy):
    def __init__(self, data_dir="./data", k=200, temperature=0.1, **kwargs):
        super().__init__(**kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.k = k
        self.temperature = temperature

        # Same encoder architecture as the clients’ encoder
        self.eval_model = ResNet18Projv2(emb_dim=2048).to(self.device)

        self.train_ld, self.test_ld = build_eval_loaders(
            data_dir=data_dir, batch_size=512, num_workers=2
        )

    def _load_encoder_from_ndarrays(self, nds):
        """
        nds contains [encoder params..., predictor params...]
        We only load the first self.n_model_params arrays into eval_model.
        """
        enc_nds = nds[: self.n_model_params]

        # Convert list[np.ndarray] -> list[torch.Tensor]
        enc_tensors = [torch.from_numpy(a) for a in enc_nds]

        # Assign in the same order as model.parameters()
        with torch.no_grad():
            for p, t in zip(self.eval_model.parameters(), enc_tensors):
                p.copy_(t.to(dtype=p.dtype).view_as(p))

    def evaluate(self, server_round, parameters):
        if parameters is None:
            return None

        nds = parameters_to_ndarrays(parameters)
        self._load_encoder_from_ndarrays(nds)

        train_feats, train_labels = extract_feats(self.eval_model, self.train_ld, self.device)
        test_feats,  test_labels  = extract_feats(self.eval_model, self.test_ld,  self.device)

        acc = knn_acc(
            train_feats, train_labels,
            test_feats,  test_labels,
            k=self.k, temperature=self.temperature
        )

        # Flower expects (loss, metrics). Use 1-acc as a “loss”.
        loss = 1.0 - acc
        return loss, {"knn_acc": acc}
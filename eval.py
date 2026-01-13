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
    """Compute ||a - b||_2 across a list of parameter tensors."""
    s = 0.0
    for ai, bi in zip(a, b):
        diff = ai.astype(np.float64) - bi.astype(np.float64)
        s += float(np.sum(diff * diff))
    return float(np.sqrt(s))




class FedAvgWithKnnEval(fl.server.strategy.FedAvg):
    def __init__(self, *args, data_dir="./data", device="cpu", **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device(device)
        self.eval_model = ResNet18Projv2(emb_dim=128).to(self.device)

        self.train_ld, self.test_ld = build_eval_loaders(
            data_dir=data_dir, batch_size=512, num_workers=2
        )

        # Cache labels/features per eval call? simplest is recompute each round.

    def _set_model_from_global_params(self, ndarrays):
        # You are federating (model params + predictor params).
        # For eval you only need the model params.
        n_model = sum(1 for _ in self.eval_model.parameters())
        model_params = ndarrays[:n_model]
        with torch.no_grad():
            for p, w in zip(self.eval_model.parameters(), model_params):
                p.copy_(torch.from_numpy(w).to(p.device))

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)

        # Flower may return (None, metrics) if nothing aggregated
        if aggregated is None:
            return None

        params_agg, metrics_agg = aggregated
        if params_agg is None:
            # Nothing to aggregate this round -> propagate None upward
            # (also prevents your autoscaler from crashing)
            return None, metrics_agg

        # --- FedEMA autoscaler code below ---
        global_nd = parameters_to_ndarrays(params_agg)
        global_model = global_nd[: self.n_model_params]

        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            if self.state.lambda_k.get(cid, None) is not None:
                continue

            client_nd = parameters_to_ndarrays(fit_res.parameters)
            client_model = client_nd[: self.n_model_params]

            div = l2_norm_between(global_model, client_model)
            lam = self.state.tau / (div + self.state.eps)
            self.state.lambda_k[cid] = float(lam)

        return params_agg, metrics_agg


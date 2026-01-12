import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from typing import Dict, Tuple, List,Optional
import flwr as fl
from flwr.common import Metrics, Context, Parameters, FitIns, EvaluateIns
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from pathlib import Path
import torch
import json
from torchvision import transforms
import numpy as np
DATA_DIR = Path(__file__).resolve().parent / "data"   # absolute path

class mseFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, topk,selection_rounds, **kwargs):
        super().__init__(**kwargs)
        self.topk = topk
        self.selection_rounds = selection_rounds
        self.selected_cids: set[str] | None = None
        self.selected_classes=[]
        self.latest_parameters: Optional[Parameters] = None


    def aggregate_fit(self, server_round, results, failures):
        # Log something you can actually see
        print(f"[server] aggregate_fit round={server_round} results={len(results)}", flush=True)

        if server_round <= self.selection_rounds:
            # results: List[Tuple[ClientProxy, FitRes]]
            qualities = []
            q_print=[]
            for client_proxy, fit_res in results:
                cid = client_proxy.cid
                q = fit_res.metrics.get("quality", None)
                #nl = fit_res.metrics.get("noise_level", None)
                nl = json.loads(fit_res.metrics.get("classes", "[]"))
                if q is not None:
                    qualities.append((cid, float(q)))
                    q_print.append((nl, round(q,3)))

            print(sorted(q_print,key=lambda x: x[1]))
            if not qualities:
                print("[server] No quality metrics returned in round 1", flush=True)

            # Lower MSE is better → sort ASC
            qualities.sort(key=lambda x: x[1], reverse=True)
            q_print.sort(key=lambda x: x[1], reverse=True)
            k = min(self.topk, len(qualities))
            #self.selected_cids = {cid for cid, _ in qualities[:k]}
            if self.selected_cids is None:
                self.selected_cids = {cid for cid, _ in qualities[:k]}
            else:
                # accumulate new ones in later rounds
                self.selected_cids.update(cid for cid, _ in qualities[:k])
            selected_noise = {tuple(nl) for nl, q in q_print[:k]}
            for i in range(len(q_print[:k])):
                self.selected_classes.append((server_round, q_print[:k][i][0]))
            noise_level=[]
            for i in selected_noise:
                noise_level.append(i)
            print(f"[server] Selected (top-{k} by lowest MSE): {noise_level}", flush=True)

            # Optional: relax min_* after selection so later rounds don't block on all clients
            self.min_available_clients = len(self.selected_cids)
            self.min_fit_clients = len(self.selected_cids)
            self.min_evaluate_clients = min(getattr(self, "min_evaluate_clients", 0) or 0, len(self.selected_cids)) or 0
            self.fraction_fit = 1.0  # schedule all selected each round

        aggregated, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            self.latest_parameters = aggregated  # <-- keep the newest global params
        return aggregated, metrics

    def configure_fit(self, server_round, parameters, client_manager):
        if server_round > self.selection_rounds and self.selected_cids:
            # Build the round config like FedAvg would
            cfg = {}
            if self.on_fit_config_fn is not None:
                cfg = self.on_fit_config_fn(server_round)
            fit_ins = FitIns(parameters, cfg)

            # Pick only selected clients that are currently available
            pool = list(client_manager.all().values())  # Dict[str, ClientProxy] -> values()
            chosen = [c for c in pool if c.cid in self.selected_cids]

            print(f"[server] configure_fit round={server_round} choosing {len(chosen)} clients", flush=True)
            return [(c, fit_ins) for c in chosen]  # <-- correct return type

        # Round 1 (or if selection missing): fall back to FedAvg behavior
        return super().configure_fit(server_round, parameters, client_manager)
    
    def configure_evaluate(self, server_round, parameters, client_manager):
        # Ensure evaluation also only uses the selected clients (after round 1)
        if server_round >=self.selection_rounds and self.selected_cids:
            cfg = self.on_evaluate_config_fn(server_round) if hasattr(self, "on_evaluate_config_fn") and self.on_evaluate_config_fn else {}
            eval_ins = EvaluateIns(parameters, cfg)
            chosen = [c for c in client_manager.all().values() if c.cid in self.selected_cids]
            return [(c, eval_ins) for c in chosen]
        return super().configure_evaluate(server_round, parameters, client_manager)
    
    def aggregate_evaluate(self, server_round, results, failures):
        loss_aggr, _ = super().aggregate_evaluate(server_round, results, failures)

        eval_pairs = [(r.num_examples, r.metrics or {}) for _, r in results]
        def weighted_avg(pairs):
            tot = sum(n for n, _ in pairs) or 1
            keys = {k for _n, m in pairs for k in m.keys()}
            return {k: sum(m.get(k, 0.0) * n for n, m in pairs) / tot for k in keys}

        agg_metrics = weighted_avg(eval_pairs)
        print(self.selected_classes)
        return loss_aggr, agg_metrics





class NoiseTransforms:
    MEAN = (0.5, 0.5, 0.5)
    STD  = (0.5, 0.5, 0.5)

    @staticmethod
    def gaussian(x: torch.Tensor, std: float) -> torch.Tensor:
        if std <= 0:
            return x
        return (x + torch.randn_like(x) * std).clamp(0.0, 1.0)

    @staticmethod
    def make_client_transform(noise_level: int):
        # noise_level ∈ [0,100]
        max_std = 0.30
        std = max(0.0, min(1.0, noise_level / 100.0)) * max_std*5
        print("applied noise: ", std, flush=True)
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: NoiseTransforms.gaussian(x, std)),
            transforms.Normalize(NoiseTransforms.MEAN, NoiseTransforms.STD),
        ])

    @staticmethod
    def make_val_transform():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(NoiseTransforms.MEAN, NoiseTransforms.STD),
        ])



class augmentTransforms:
    MEAN = (0.5, 0.5, 0.5)
    STD  = (0.5, 0.5, 0.5)
    @staticmethod
    def gaussian(x: torch.Tensor, std: float) -> torch.Tensor:
        if std <= 0:
            return x
        return (x + torch.randn_like(x) * std).clamp(0.0, 1.0)

    @staticmethod
    def make_train_transform():
        t = []
        t += [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(p=0.5),
            # Optional extras (uncomment if you want stronger aug):
            # transforms.ColorJitter(0.2,0.2,0.2,0.1),
            # transforms.RandAugment(num_ops=2, magnitude=9),
        ]
        t += [transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
        t += [transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3))]
        return transforms.Compose([
            t,
        ])
    
    def make_train_transform2():
        NORM = transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  # or dataset stats
        ops = [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            NORM,
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
        ]
        return transforms.Compose(ops)

    @staticmethod
    def make_val_transform():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(NoiseTransforms.MEAN, NoiseTransforms.STD),
        ])








class MyFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, topk, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # cache of last-known client metrics by cid
        self.client_meta: Dict[str, Metrics] = {}
        self.topk=topk

    # Capture metrics after each round of fit
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[Tuple[ClientProxy, fl.common.FitRes]],
    ):
        # cache per-client metrics
        for client, fitres in results:
            self.client_meta[client.cid] = {
                **self.client_meta.get(client.cid, {}),
                **(fitres.metrics or {}),
            }

        # Let FedAvg do param aggregation
        return super().aggregate_fit(server_round, results, failures)
    
    def aggregate_evaluate(self, server_round, results, failures):
        loss_aggr, _ = super().aggregate_evaluate(server_round, results, failures)

        eval_pairs = [(r.num_examples, r.metrics or {}) for _, r in results]
        def weighted_avg(pairs):
            tot = sum(n for n, _ in pairs) or 1
            keys = {k for _n, m in pairs for k in m.keys()}
            return {k: sum(m.get(k, 0.0) * n for n, m in pairs) / tot for k in keys}

        agg_metrics = weighted_avg(eval_pairs)

        return loss_aggr, agg_metrics

    # Choose exactly which clients to train next round
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        all_clients = list(client_manager.all().values())

        # If we don't have custom metrics yet (e.g., round 1), fall back to everyone
        if not self.client_meta:
            selected = all_clients
        else:
            # Sort by the last known "custom" metric (descending). Missing -> -inf
            def custom_of(c: ClientProxy) -> float:
                m = self.client_meta.get(c.cid, {})
                #print(float(m.get("shannon", float("-inf"))))
                return float(m.get("shannon", float("-inf")))
    
            selected = sorted(all_clients, key=custom_of, reverse=True)[: self.topk]
        # Build FitIns (reuse any per-round config if present)
        cfg = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
        fit_ins = FitIns(parameters, cfg)
        return [(c, fit_ins) for c in selected]
    
    
class vqvae_utils():
    @staticmethod
    def _normalize(p, eps=1e-12):
        p = np.clip(p, 0.0, None)
        s = p.sum()
        return p / (s + eps)
    @staticmethod
    def js_divergence(p, q, eps=1e-12):
        """Jensen–Shannon divergence (symmetric, bounded, stable)."""
        p = vqvae_utils._normalize(p, eps); q =vqvae_utils._normalize(q, eps)
        m = 0.5 * (p + q)
        kl_pm = np.sum(p * (np.log(p + eps) - np.log(m + eps)))
        kl_qm = np.sum(q * (np.log(q + eps) - np.log(m + eps)))
        return 0.5 * (kl_pm + kl_qm)
    @staticmethod
    def cosine_distance(p, q, eps=1e-12):
        p = vqvae_utils._normalize(p, eps); q = vqvae_utils._normalize(q, eps)
        num = np.dot(p, q)
        den = (np.linalg.norm(p) * np.linalg.norm(q)) + eps
        return 1.0 - (num / den)
    @staticmethod
    def utility_neg_js(p_i, p_target):
        """Higher is better → negate the divergence."""
        return -vqvae_utils.js_divergence(p_i, p_target)
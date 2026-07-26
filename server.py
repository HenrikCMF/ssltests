# fedema_strategy.py
from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import flwr as fl
from flwr.common import FitIns, Parameters, ndarrays_to_parameters, parameters_to_ndarrays, bytes_to_ndarray
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms as T
import flwr as fl
from flwr.common import parameters_to_ndarrays
from dataloader import CIFAR10BYOLClientData, CIFAR100BYOLClientData, TinyImageNetBYOLClientData

import numpy as np
from typing import List
from attacks import Loki, LokiConfig

# Throughput flags for the server-side kNN eval (full forward over CIFAR each round).
#torch.set_float32_matmul_precision("high")
#torch.backends.cuda.matmul.allow_tf32 = True
#torch.backends.cudnn.allow_tf32 = True
#torch.backends.cudnn.benchmark = False#True


def rss_gb() -> float:
    """Resident host memory (GB) of the *current* process, read from /proc."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024 * 1024)   # kB -> GB
    except Exception:
        pass
    return -1.0


def log_mem(tag: str) -> None:
    """DIAGNOSTIC: print this process's RSS (+ CUDA reserved if any) for tag."""
    cuda = ""
    if torch.cuda.is_available():
        cuda = (f"  cuda_alloc={torch.cuda.memory_allocated()/1e9:.2f}GB"
                f" reserved={torch.cuda.memory_reserved()/1e9:.2f}GB")
    print(f"[MEM] {tag}: rss={rss_gb():.2f}GB{cuda}", flush=True)

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
    use_amp = (str(device).startswith("cuda") or getattr(device, "type", "") == "cuda")
    for x, y in loader:
        x = x.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            # Paper eval: remove encoder MLP -> use backbone representation
            if use_pred:
                z = encoder(x, normalize=False)
                if normalize:
                    z = F.normalize(z, dim=1)
            else:
                z = encoder.encode_backbone(x, normalize=normalize)  # [B,512]

        # Keep on device so the kNN similarity matmul runs on GPU.
        feats.append(z.float().detach())
        labels.append(y.to(device, non_blocking=True))
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
            lam = self.state.tau / (div + self.state.eps)
            #lam=1e-8
            self.state.lambda_k[cid] = float(lam)

        return params_agg, metrics_agg

class FedEMAStrategyWithKnn(FedEMAStrategy):
    def __init__(self, data_dir="./data", dataset: str = "cifar10",
                 k=200, temperature=0.1, eval_model=None,
                 loki_config: Optional[LokiConfig] = None, loki_enabled: bool = False,
                 loki_target_cid: int = 0, loki_extract_all: bool = False,
                 loki_save_fragments: bool = True,
                 loki_fragments_dir: str = "fragments",
                 **kwargs):
        super().__init__(**kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "mps" if torch.backends.mps.is_available() else "cpu")
        self.k = k
        self.temperature = temperature

        # --- LOKI in-the-loop attack state (see attacks.Loki) ---------------- #
        # When enabled, the malicious server arms a trap module for the target
        # client only (model inconsistency) via configure_fit, then reconstructs
        # that client's images from its returned update in aggregate_fit, re-arms,
        # and repeats. All of this is inert when loki_enabled is False, so a
        # normal FedBYOL/FedEMA run is unchanged.
        self.loki_enabled = loki_enabled
        self.loki = Loki(loki_config) if loki_enabled else None
        self.loki_target_cid = loki_target_cid
        # Full model inconsistency: arm every client with its own identity mapping
        # set and reconstruct all of them from the de-aggregated aggregate update
        # (Sec. 3.5/4.2). When False, only loki_target_cid is armed/reconstructed.
        self.loki_extract_all = loki_extract_all
        # Fragment dumping: per-bin .pt stacks for offline per-image reconstruction.
        self.loki_save_fragments = loki_save_fragments
        self.loki_fragments_dir = loki_fragments_dir
        # Flower hands the strategy opaque ClientProxy ids; clients report their
        # partition id in fit metrics so we can find the target (built round 1).
        self._proxy_to_partition: Dict[str, int] = {}
        # The FC1 weight/bias we armed the target with this round (to diff against
        # the returned update). None until the target has been armed. In all-client
        # mode the binning FC1 is identical for every client (only the conv set
        # differs), so a single armed copy still diffs every client's update.
        self._armed_fc1 = None
        self._armed_bias = None
        # Partitions armed this round (all-client mode); reconstructed in aggregate_fit.
        self._armed_cids: list = []

        # Same encoder architecture as the clients’ encoder
        self.eval_model = eval_model

        _eval_cls = {
            "tiny_imagenet": TinyImageNetBYOLClientData,
            "cifar100": CIFAR100BYOLClientData,
        }.get(dataset, CIFAR10BYOLClientData)
        self.train_ld, self.test_ld = _eval_cls.build_eval_loaders(
            data_dir=data_dir, batch_size=128, num_workers=1
        )
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

    # ------------------------------------------------------------------ #
    # LOKI in-the-loop attack
    # ------------------------------------------------------------------ #
    def configure_fit(self, server_round, parameters, client_manager):
        # Start from the FedEMA configuration (per-client lambda/selection config).
        pairs = super().configure_fit(server_round, parameters, client_manager)
        if not self.loki_enabled:
            return pairs

        if self.loki_extract_all:
            return self._configure_fit_all(server_round, pairs)

        # Model inconsistency: send the target client an armed trap module and
        # every other client an inert (all-zero) module so they train the benign
        # network untouched. Until the proxy->partition map is known (round 1),
        # everyone is inert and the attack starts once the target is identified.
        #
        # Previous version deserialized the (identical) global model once PER
        # client and built fresh all-zero fc1/fc2 (~0.5 GB each) for every inert
        # client — ~10x the model size in transient host RAM. Kept for recovery:
        # armed = []
        # for client, fitins in pairs:
        #     nds = parameters_to_ndarrays(fitins.parameters)
        #     partition = self._proxy_to_partition.get(client.cid, None)
        #     if partition == self.loki_target_cid:
        #         nds = self._arm_target_module(nds, server_round)
        #     else:
        #         nds = self._inert_module(nds)
        #     armed.append((client, FitIns(ndarrays_to_parameters(nds), fitins.config)))
        # return armed

        # The parent broadcasts the SAME global parameters to every client (only
        # the per-client config differs), so deserialize it once and build just
        # two payloads: one armed (target) and one inert (everyone else). Every
        # non-target client is byte-identical, so they share a single serialized
        # Parameters object instead of one per client. Both payloads also reuse
        # the unchanged backbone arrays by reference (only the 4 module slots
        # differ). This cuts configure_fit's peak host RAM several-fold.
        base_nds = parameters_to_ndarrays(pairs[0][1].parameters)
        armed_params = None
        inert_params = None
        out = []
        for client, fitins in pairs:
            partition = self._proxy_to_partition.get(client.cid, None)
            if partition == self.loki_target_cid:
                if armed_params is None:
                    armed_params = ndarrays_to_parameters(
                        self._arm_target_module(base_nds, server_round))
                out.append((client, FitIns(armed_params, fitins.config)))
            else:
                if inert_params is None:
                    inert_params = ndarrays_to_parameters(self._inert_module(base_nds))
                out.append((client, FitIns(inert_params, fitins.config)))
        return out

    def _configure_fit_all(self, server_round, pairs):
        """
        Full model inconsistency (Sec. 3.5/4.2): arm every client with its OWN
        identity mapping set so each client's images land in a disjoint block of
        FC1's input, recoverable from the aggregate. The binning FC1/bias and FC2
        are identical for all clients (same brightness cutoffs); only the conv
        identity set differs per client, so the big FC1/FC2 arrays are built once
        and shared by reference, while only the tiny conv slot is rebuilt per cid.

        Clients whose partition id is not yet known (round 1, before any fit
        metrics arrive) are sent an inert module and skipped this round.
        """
        base_nds = parameters_to_ndarrays(pairs[0][1].parameters)
        ci, w1, b1, w2 = self._module_slots(base_nds)

        # Redraw cutoffs from the learned distribution, then build the shared
        # binning module once (conv set is overwritten per client below).
        self.loki.setup_fc_biases("agnostic")
        module = self.loki.build_module(active_cid=0)
        w1_arr = module.fc1.weight.detach().cpu().numpy().astype(base_nds[w1].dtype)
        b1_arr = module.fc1.bias.detach().cpu().numpy().astype(base_nds[b1].dtype)
        w2_arr = module.fc2.weight.detach().cpu().numpy().astype(base_nds[w2].dtype)
        # One armed FC1/bias copy diffs every client's update (binning is shared).
        self._armed_fc1 = w1_arr.copy()
        self._armed_bias = b1_arr.copy()

        self._armed_cids = []
        armed_cache = {}            # partition -> serialized Parameters
        inert_params = None
        out = []
        for client, fitins in pairs:
            partition = self._proxy_to_partition.get(client.cid, None)
            if partition is None:
                if inert_params is None:
                    inert_params = ndarrays_to_parameters(self._inert_module(base_nds))
                out.append((client, FitIns(inert_params, fitins.config)))
                continue
            if partition not in armed_cache:
                # Per-client identity mapping set: only the conv slot differs.
                self.loki._set_identity_mapping(module, partition)
                conv_arr = module.conv.weight.detach().cpu().numpy().astype(base_nds[ci].dtype)
                nds = list(base_nds)
                nds[ci], nds[w1], nds[b1], nds[w2] = conv_arr, w1_arr, b1_arr, w2_arr
                armed_cache[partition] = ndarrays_to_parameters(nds)
                self._armed_cids.append(partition)
            out.append((client, FitIns(armed_cache[partition], fitins.config)))
        return out

    def aggregate_fit(self, server_round, results, failures):
        if self.loki_enabled:
            # Refresh the proxy->partition map from the clients' reported ids.
            for client, fit_res in results:
                pid = (fit_res.metrics or {}).get("cid")
                if pid is not None:
                    self._proxy_to_partition[client.cid] = int(pid)
            if self.loki_extract_all:
                # Reconstruct every armed client from the de-aggregated aggregate.
                if self._armed_fc1 is not None and self._armed_cids:
                    self._loki_reconstruct_all(server_round, results)
            elif self._armed_fc1 is not None:
                # Single target: reconstruct from its individual update.
                for client, fit_res in results:
                    if self._proxy_to_partition.get(client.cid) == self.loki_target_cid:
                        self._loki_reconstruct(server_round, fit_res)
                        break
        agg = super().aggregate_fit(server_round, results, failures)
        # DIAGNOSTIC: force collection then report server RSS so a steady climb
        # (true leak) is distinguishable from GC lag. Remove once memory is traced.
        gc.collect()
        log_mem(f"server aggregate_fit r{server_round}")
        return agg

    def _module_slots(self, nds):
        """Indices of the LOKI module's float arrays (they lead the encoder)."""
        # LokiEncoder.state_dict order: loki.conv.weight, loki.fc1.weight,
        # loki.fc1.bias, loki.fc2.weight, then the benign net.
        return 0, 1, 2, 3  # conv.weight, fc1.weight, fc1.bias, fc2.weight

    def _arm_target_module(self, nds, server_round):
        """Overwrite the module slots with trap weights crafted for the target."""
        nds = list(nds)
        # Re-draw bias cutoffs from the (incrementally learned, Sec. 4.1)
        # distribution each round, then build the binning/identity-map module.
        self.loki.setup_fc_biases("agnostic")
        module = self.loki.build_module(active_cid=0)
        ci, w1, b1, w2 = self._module_slots(nds)
        nds[ci] = module.conv.weight.detach().cpu().numpy().astype(nds[ci].dtype)
        nds[w1] = module.fc1.weight.detach().cpu().numpy().astype(nds[w1].dtype)
        nds[b1] = module.fc1.bias.detach().cpu().numpy().astype(nds[b1].dtype)
        nds[w2] = module.fc2.weight.detach().cpu().numpy().astype(nds[w2].dtype)
        # Remember what we sent so we can diff it against the returned update.
        self._armed_fc1 = nds[w1].copy()
        self._armed_bias = nds[b1].copy()
        return nds

    def _inert_module(self, nds):
        """Zero the module slots -> a transparent, zero-gradient passthrough."""
        nds = list(nds)
        for i in self._module_slots(nds):
            nds[i] = np.zeros_like(nds[i])
        return nds

    def _loki_reconstruct(self, server_round, fit_res):
        # Only FC1's weight/bias slots are read from the update, so decode just
        # those two tensors instead of materializing the whole client model:
        # the FC1 weight alone is multi-GB, so deserializing every layer (and
        # holding a separate diff array) is what spikes server RSS each round.
        _, w1, b1, _ = self._module_slots(None)
        tensors = fit_res.parameters.tensors
        ret_w1 = bytes_to_ndarray(tensors[w1])
        ret_b1 = bytes_to_ndarray(tensors[b1])
        # Eq. 6: the update the server sees is Theta_t - Theta_{t+1} (armed - returned).
        # Subtract the returned weights into the armed copy in place so no third
        # multi-GB array is allocated; from_numpy then aliases that same buffer.
        self._armed_fc1 -= ret_w1
        self._armed_bias -= ret_b1
        upd_w = torch.from_numpy(self._armed_fc1)
        upd_b = torch.from_numpy(self._armed_bias)
        # The diff now lives in upd_w/upd_b; release the armed copies (re-armed
        # fresh next round, so not needed here). None — not del — keeps the
        # `is not None` guard in aggregate_fit valid on rounds that don't arm.
        self._armed_fc1 = None
        self._armed_bias = None
        frags, counts = self.loki.reconstruct_with_bias_count(upd_w, upd_b, target_cid=0)

        # LEAK_NOISE mapping inputs, both read from the FC1 WEIGHT update LOKI
        # actually reconstructs from (upd_w) -- NOT the client's logged
        # eff.abs.noise_std, which is for the dominant-NORM layer (the FC1 bias /
        # conv, d=fc_size), a different layer LOKI never reads.
        #   fc1w_rms = ||upd_w|| / sqrt(numel): RMS of the weight update. The DP
        #     noise added to THIS layer had absolute std ~= sigma * fc1w_rms.
        #   m_fired  = max-abs of each fired bin's Eq.9 source row (the per-bin
        #     denominator reconstruct() divides by). The clean signal, not noise.
        # After Eq.9 (divide row by its m_fired), bin i's fragment noise std is
        # (sigma * fc1w_rms) / m_fired[i]. This is the SINGLE-target path (upd_w is
        # one client's update), so no cross-client sqrt(K):
        #   LEAK_NOISE ~= sigma * fc1w_rms / median(m_fired)   [num_clients=1]
        # (reconstruction_test.dp_noise_to_leak_noise; the sqrt(K) factor there is
        # only for the aggregate _loki_reconstruct_all path). NOTE m_fired spans
        # ~100x across bins, so one scalar LEAK_NOISE only matches the MEDIAN bin.
        # bin_mass is frag-free to stay within the server's memory budget.
        _mass = self.loki.bin_mass(upd_w, target_cid=0)
        _mf = _mass[self.loki.fired_mask(_mass)]
        if _mf.numel():
            _rms = float(upd_w.norm()) / (upd_w.numel() ** 0.5)
            print(f"r{int(server_round)} loki(cid0): fc1w_rms={_rms:.4g}  "
                  f"m_fired median={_mf.median():.4g} min={_mf.min():.4g} "
                  f"max={_mf.max():.4g} n_fired={_mf.numel()}  "
                  f"[single-target: LEAK_NOISE ~= DP_NOISE*fc1w_rms/m_fired]")
        del _mass, _mf

        # Incremental bias learning (Sec. 4.1): refine the distribution using the
        # cutoffs of every *fired* bin (count != 0), not just clean single-image
        # bins. A fired neuron's cutoff sits just below a real image's brightness,
        # so fired cutoffs sample the brightness density directly; blends sit in
        # high-density regions and are correctly weighted. The clean-only set is a
        # biased, under-dispersed sub-population (it selects the lowest-gradient
        # bins) and shrinks the learned std, which bunches next round's cutoffs and
        # compounds into lower leakage.
        fired = counts != 0
        if bool(fired.any()):
            # `counts` is aligned 1:1 with the per-bin cutoff that owns it:
            # FedAVG counts are per-neuron (full length); FedSGD counts are
            # consecutive-bin differences, so they map to the lower cutoff edge.
            cutoffs = self.loki.cutoffs
            if not self.loki.cfg.fedavg:
                cutoffs = cutoffs[:-1]
            cutoffs = cutoffs.to(fired.device)
            self.loki.learn_bias_distribution(cutoffs[fired].cpu().tolist())

        # Dump per-bin fragments so each (probable) original image can be
        # reconstructed offline from only its own across-round augmented views.
        if self.loki_save_fragments:
            # One file per round (regroup re-clusters across bins anyway, so the
            # per-bin layout is unnecessary and floods the disk with tiny files).
            # Previous per-bin saver kept commented for easy recovery:
            # self.loki.save_fragments_by_bin(
            #     frags, self.loki_fragments_dir, server_round, counts=counts
            # )
            self.loki.save_round_fragments(
                frags, self.loki_fragments_dir, server_round, counts=counts
            )

        n_frags = len(frags)
        print(f"[LOKI] round {int(server_round)}: reconstructed {n_frags} fragments")

        # Drop the round's large transients before returning so the next round
        # doesn't stack on top of them; the gc.collect() in aggregate_fit then
        # reports the true resident set.
        del upd_w, upd_b, ret_w1, ret_b1, frags, counts
        gc.collect()

    def _loki_reconstruct_all(self, server_round, results):
        """
        Full model inconsistency reconstruction (Sec. 3.5/4.2): form the secure
        aggregate of the armed clients' FC1 weight updates and de-aggregate every
        client from its own block.

        The binning FC1 sent was identical for all clients, so the aggregated
        weight "gradient" (Eq. 6 with secure-agg = sum) is n*armed - sum(returned).
        Each client's images sit in a disjoint block of FC1's input (its identity
        mapping set), and a client that did not own a block contributes ~0 there,
        so reconstructing block k recovers client k. The FC1 *bias* gradient is
        coupled across clients by aggregation and so cannot give a per-client
        count; the per-client fired/empty signal is therefore read from the
        de-aggregated *weight* block (reconstruct_with_fired).
        """
        _, w1, _, _ = self._module_slots(None)
        armed = set(self._armed_cids)

        # Securely aggregate (sum) the armed clients' returned FC1 weights.
        sum_w = None
        n = 0
        for client, fit_res in results:
            if self._proxy_to_partition.get(client.cid) not in armed:
                continue
            rw = bytes_to_ndarray(fit_res.parameters.tensors[w1])
            if sum_w is None:
                sum_w = rw                       # take ownership of the first array
            else:
                sum_w += rw
            n += 1
        if sum_w is None:
            self._armed_fc1 = None
            self._armed_bias = None
            return

        # Eq. 6 aggregated weight gradient: n*armed - sum(returned), in place to
        # avoid a third multi-GB array.
        self._armed_fc1 *= n
        self._armed_fc1 -= sum_w
        upd_w = torch.from_numpy(self._armed_fc1)
        del sum_w
        self._armed_fc1 = None
        self._armed_bias = None

        cutoffs = self.loki.cutoffs
        if not self.loki.cfg.fedavg:
            cutoffs = cutoffs[:-1]

        fired_cutoffs = []
        total = 0
        for cid in sorted(armed):
            frags, mass = self.loki.reconstruct_with_fired(upd_w, target_cid=cid)
            fired = self.loki.fired_mask(mass)
            if bool(fired.any()):
                fired_cutoffs.extend(cutoffs.to(fired.device)[fired].cpu().tolist())
                if cid == 0:
                    # m_fired = median max-abs of a fired bin's Eq.9 source row (the
                    # per-bin normalizer). Paired with the client's logged DP
                    # eff.abs.noise_std, this pins the sandbox LEAK_NOISE equivalent:
                    # LEAK_NOISE ~= sqrt(K) * eff_std / m_fired  (see reconstruction_test
                    # .dp_noise_to_leak_noise). Logged because RMS(eff_std) << peak(m_fired)
                    # is exactly the sparsity confound that makes equal values diverge.
                    fm = mass[fired]
                    print(f"r{int(server_round)} loki(cid0): m_fired median={fm.median():.4g} "
                          f"min={fm.min():.4g} max={fm.max():.4g}  n_fired={int(fired.sum())} "
                          f"[Eq.9 denom; use with dp eff_std -> LEAK_NOISE]")
            if self.loki_save_fragments:
                # Per-client subdir so reconstructed data stays tied to its owner
                # (Sec. 4.2). counts=fired marks the kept (non-empty) bins.
                cdir = os.path.join(self.loki_fragments_dir, f"client_{cid:02d}")
                self.loki.save_round_fragments(
                    frags, cdir, server_round, counts=fired.long()
                )
            total += int(fired.sum())
            del frags, mass, fired

        # Bias-distribution learning from every fired bin across all clients
        # (the binning FC1/bias is shared, so the distribution is global).
        if fired_cutoffs:
            self.loki.learn_bias_distribution(fired_cutoffs)

        print(f"[LOKI] round {int(server_round)}: reconstructed {total} fragments "
              f"across {len(armed)} clients")
        del upd_w
        gc.collect()

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
        if server_round % 5 != 0:
            return None
        if parameters is None:
            return None

        nds = parameters_to_ndarrays(parameters)
        if len(nds) == 0:
            return None


        if self.loki_enabled:
            nds = self._inert_module(nds)

        self._load_encoder_from_ndarrays(nds)
        #calibrate_bn(self.eval_model, self.train_ld, self.device, num_batches=50)
        # Extract features once (backbone 512-dim, same as kNN)
        train_feats, train_labels = extract_feats(self.eval_model, self.train_ld, self.device, normalize=True, use_pred=False)
        test_feats,  test_labels  = extract_feats(self.eval_model, self.test_ld,  self.device, normalize=True, use_pred=False)


        # kNN (existing)
        #k_eff = min(self.k, train_feats.shape[0])
        b_acc_knn = knn_acc(
            train_feats, train_labels,
            test_feats,  test_labels,
            k=self.k, temperature=self.temperature
        )

        loss = 1.0 - b_acc_knn  # Flower still expects a loss; you can change to 1-acc_linear if you prefer

        # Save the BEST-knn encoder, not the latest. Under DP noise BYOL suffers
        # representation collapse -- a sudden, sustained fall to ~10% knn (random on
        # CIFAR-10) -- whose onset comes EARLIER the more noise there is, so the
        # FINAL-round model is collapsed at every eps in a DP sweep. Downstream LOKI
        # reconstruction clusters the leaked fragments with THIS encoder; a collapsed
        # encoder maps every fragment to ~one point and clustering (hence the
        # "fraction reconstructed well" metric) is destroyed. Gating the save on a
        # knn improvement keeps eval_model.pth at the peak encoder at run end. The
        # peak round matches the driver's peak_knn, so model and reported accuracy
        # come from the same round.
        is_best = b_acc_knn > getattr(self, "_best_knn_acc", -1.0)
        if is_best:
            self._best_knn_acc = b_acc_knn
            torch.save(self.eval_model.state_dict(), "eval_model.pth")

        # Training-quality axis (pair with the LOKI m_fired leak axis) so a DP sweep
        # shows both effects in one log.
        print(f"r{int(server_round)} eval: knn_acc={b_acc_knn*100:.2f}%"
              f"{'  [NEW BEST -> eval_model.pth]' if is_best else ''}  [training-quality axis]")

        # The 50k feature bank + the 10k x 50k kNN similarity matmul are sizeable
        # GPU allocations that PyTorch's caching allocator otherwise keeps reserved
        # across rounds. The driver shares one physical GPU with the client actor
        # under Flower simulation, so drop the tensors and hand the reservation back
        # before the next round of client training starts.
        del train_feats, train_labels, test_feats, test_labels
        # eval_model carries the full ~3 GB LOKI trap; extract_feats moved it onto
        # the GPU the client actor trains on. It's idle until the next eval (every
        # 5 rounds), so park it back on CPU so it stops squatting VRAM between evals
        # (this was the ~3.5 GB second process pinned on the card).
        self.eval_model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return loss, {"knn_acc": b_acc_knn}#, "pred_knn_acc": pred_acc_knn}#, "linear_acc": acc_linear}
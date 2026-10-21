"""Callable aggregators that replace Flower's FedAvg.

``RandomAggregator`` draws trimmed mean / winsorized mean / Huber M-estimator.
``DistributionAggregator`` draws each coordinate from the middle of a Gaussian
fit to the client values.

FedAvg is still implemented (pass ``methods=("fedavg", ...)``) but is not in the
default mix: a linear round lets LOKI through. Flower's trimmed mean
(``FedTrimmedAvg``) stacks every client's full layer, which OOMs on the LOKI FC1.
Winsorized mean and M-estimators are not in Flower.

Coordinate-wise work runs in PyTorch on CPU with layout (n, K). NumPy
``median``/``partition`` along axis 0 of (K, n) is ~10x slower, and the previous
Huber path (10 IRLS rounds, MAD re-estimated every round, boolean fancy
indexing) was ~300x FedAvg on the trap FC1 — enough to nearly double a round.

Use as an object::

    agg = RandomAggregator()
    global_nd = agg(results)          # draws a method, then reduces
    print(agg.last_method, agg.last_elapsed)
"""
from __future__ import annotations

import time
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from flwr.common import FitRes, bytes_to_ndarray
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate_inplace

ClientResults = Sequence[Tuple[ClientProxy, FitRes]]

# Huber k ≈ 95% efficiency at the Gaussian. Scale is MAD / 0.6745.
_HUBER_C = 1.345
_MAD_TO_STD = 0.67448975
# 5 clients * 2e6 * 4 B ≈ 40 MB of stacked scratch per chunk.
_CHUNK = 2_000_000
# sqrt(2), for Φ^{-1}(p) = sqrt(2) * erfinv(2p - 1).
_SQRT2 = 2.0 ** 0.5
# Location IRLS with MAD held at the median start. K=5 converges in 1–2 steps;
# re-estimating MAD for 10 rounds was the bulk of the old wall time.
_HUBER_ITERS = 2


class RandomAggregator:
    """Uniform draw over coordinate-wise aggregators, then reduce ``results``.

    Parameters
    ----------
    methods:
        Names from ``KNOWN``. Default is ``METHODS`` (no FedAvg).
        A one-element list is deterministic.
    seed:
        RNG seed for the per-call draw. ``None`` = nondeterministic.
    trim_beta:
        Fraction cut from each tail (Flower / scipy ``trim_mean``: ``int(p*K)``).
        With 5 clients and 0.2 this drops (or clamps) 1 client per side.
    huber_c:
        Huber threshold in units of robust scale.
    huber_iters:
        IRLS location updates after the median start (scale is not re-estimated).
    """

    KNOWN = ("fedavg", "trimmed_mean", "winsorized_mean", "m_estimator")
    METHODS = ("trimmed_mean", "winsorized_mean", "m_estimator")

    def __init__(
        self,
        methods: Sequence[str] = METHODS,
        *,
        seed: Optional[int] = None,
        trim_beta: float = 0.2,
        huber_c: float = _HUBER_C,
        huber_iters: int = _HUBER_ITERS,
        chunk_elems: int = _CHUNK,
    ) -> None:
        methods = tuple(methods)
        unknown = [m for m in methods if m not in self.KNOWN]
        if unknown:
            raise ValueError(f"unknown methods {unknown}; valid={list(self.KNOWN)}")
        if not methods:
            raise ValueError("methods must be non-empty")
        if huber_iters < 1:
            raise ValueError("huber_iters must be >= 1")
        self.methods = methods
        self.trim_beta = float(trim_beta)
        self.huber_c = float(huber_c)
        self.huber_iters = int(huber_iters)
        self.chunk_elems = int(chunk_elems)
        self._rng = np.random.default_rng(seed)
        self.last_method: Optional[str] = None
        self.last_elapsed: Optional[float] = None

        self._dispatch: dict[str, Callable[[ClientResults], List[np.ndarray]]] = {
            "fedavg": self._fedavg,
            "trimmed_mean": self._trimmed_mean,
            "winsorized_mean": self._winsorized_mean,
            "m_estimator": self._m_estimator,
        }

    def __call__(self, results: ClientResults) -> List[np.ndarray]:
        method = str(self._rng.choice(self.methods))
        self.last_method = method
        t0 = time.perf_counter()
        out = self._dispatch[method](results)
        self.last_elapsed = time.perf_counter() - t0
        return out

    def __repr__(self) -> str:
        return (
            f"RandomAggregator(methods={self.methods}, "
            f"trim_beta={self.trim_beta}, huber_c={self.huber_c}, "
            f"huber_iters={self.huber_iters})"
        )

    def _fedavg(self, results: ClientResults) -> List[np.ndarray]:
        # Flower's n_k-weighted mean (same as FedAvg with inplace=True).
        return aggregate_inplace(list(results))

    def _trimmed_mean(self, results: ClientResults) -> List[np.ndarray]:
        return self._reduce_layers(results, self._trimmed_chunk)

    def _winsorized_mean(self, results: ClientResults) -> List[np.ndarray]:
        return self._reduce_layers(results, self._winsorized_chunk)

    def _m_estimator(self, results: ClientResults) -> List[np.ndarray]:
        return self._reduce_layers(results, self._huber_chunk)

    def _reduce_layers(
        self,
        results: ClientResults,
        chunk_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> List[np.ndarray]:
        """Decode one layer at a time; reduce in ravel-chunks so FC1 is not stacked 5x."""
        n_layers = len(results[0][1].parameters.tensors)
        out: List[np.ndarray] = []
        for i in range(n_layers):
            layers = [bytes_to_ndarray(fit_res.parameters.tensors[i]) for _, fit_res in results]
            out.append(self._reduce_layer(layers, chunk_fn))
            del layers
        return out

    def _reduce_layer(
        self,
        layers: List[np.ndarray],
        chunk_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> np.ndarray:
        shape = layers[0].shape
        dtype = layers[0].dtype
        n = int(layers[0].size)
        work_dtype = np.float32 if np.issubdtype(dtype, np.floating) else np.float64
        buf = np.empty(n, dtype=work_dtype)
        flats = [np.ravel(a) for a in layers]
        for start in range(0, n, self.chunk_elems):
            end = min(start + self.chunk_elems, n)
            # Last dim = clients: torch.median/sort on (n, K) beat NumPy on (K, n).
            stacked = torch.stack(
                [torch.from_numpy(np.ascontiguousarray(f[start:end])) for f in flats],
                dim=-1,
            )
            if stacked.dtype != torch.float32 and np.issubdtype(dtype, np.floating):
                stacked = stacked.to(torch.float32)
            buf[start:end] = chunk_fn(stacked).numpy()
        return buf.reshape(shape).astype(dtype, copy=False)

    def _tail_cuts(self, n_clients: int) -> Tuple[int, int]:
        lower = int(self.trim_beta * n_clients)
        upper = n_clients - lower
        if lower > upper:
            raise ValueError(
                f"trim/winsor proportion {self.trim_beta} too large for {n_clients} clients"
            )
        return lower, upper

    def _trimmed_chunk(self, stacked: torch.Tensor) -> torch.Tensor:
        k = stacked.shape[-1]
        lower, upper = self._tail_cuts(k)
        if lower == 0:
            return stacked.mean(dim=-1)
        return stacked.sort(dim=-1).values[..., lower:upper].mean(dim=-1)

    def _winsorized_chunk(self, stacked: torch.Tensor) -> torch.Tensor:
        k = stacked.shape[-1]
        lower, upper = self._tail_cuts(k)
        if lower == 0:
            return stacked.mean(dim=-1)
        part = stacked.sort(dim=-1).values
        part[..., :lower] = part[..., lower : lower + 1]
        part[..., upper:] = part[..., upper - 1 : upper]
        return part.mean(dim=-1)

    def _huber_chunk(self, stacked: torch.Tensor) -> torch.Tensor:
        # Standard Huber location M-estimator: MAD scale from the median start,
        # then a few IRLS weighted means. Scale is not refreshed each iteration.
        mu = stacked.median(dim=-1).values
        mad = (stacked - mu.unsqueeze(-1)).abs().median(dim=-1).values
        thresh = (self.huber_c * (mad / _MAD_TO_STD).clamp_min(1e-12)).unsqueeze(-1)
        for _ in range(self.huber_iters):
            abs_r = (stacked - mu.unsqueeze(-1)).abs().clamp_min(1e-12)
            w = (thresh / abs_r).clamp_max(1.0)
            mu = (w * stacked).sum(dim=-1) / w.sum(dim=-1).clamp_min(1e-12)
        return mu


class DistributionAggregator(RandomAggregator):
    """Draw each coordinate from a Gaussian trimmed at both tails.

    The K client values at one coordinate are the samples of that weight.
    Their mean and variance (divisor K) define a Gaussian. The aggregate is
    one draw from that Gaussian with probability ``tail`` removed from each
    side, so the default 0.25 keeps the middle 50%. Every value inside the
    band can occur. Matching client values have variance 0 and the draw
    equals them.

    Same chunked CPU layout ``(n, K)`` as the other reducers: one mean, one
    variance, and a vectorized inverse CDF. No sort.
    """

    def __init__(
        self,
        *,
        seed: Optional[int] = None,
        tail: float = 0.35,#0.25,
        chunk_elems: int = _CHUNK,
    ) -> None:
        if not 0.0 < float(tail) < 0.5:
            raise ValueError(f"tail must be in (0, 0.5), got {tail}")
        # Parent validates ``methods`` against the mean-aggregator set. The
        # draw itself is ``_distribution_chunk``; ``methods`` is overwritten
        # so introspection does not claim a mean was used. ``trim_beta`` is
        # unused here.
        super().__init__(
            methods=("trimmed_mean",),
            seed=seed,
            trim_beta=0.2,
            chunk_elems=chunk_elems,
        )
        self.methods = ("distribution",)
        self.tail = float(tail)
        self.last_method = "distribution"

    def __call__(self, results: ClientResults) -> List[np.ndarray]:
        self.last_method = "distribution"
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self._reduce_layers(results, self._distribution_chunk)
        self.last_elapsed = time.perf_counter() - t0
        return out

    def __repr__(self) -> str:
        return (
            f"DistributionAggregator(tail={self.tail}, "
            f"chunk_elems={self.chunk_elems})"
        )

    def _distribution_chunk(self, stacked: torch.Tensor) -> torch.Tensor:
        # Inverse CDF of N(mu, sigma^2) restricted to probability [tail, 1-tail].
        # torch.var on this (n, K=5) layout is ~20x slower than the two-pass
        # (about 160ms vs 10ms per 2e6-row chunk on CPU).
        mu = stacked.mean(dim=-1, keepdim=True)
        sigma = (stacked - mu).square().mean(dim=-1).sqrt()
        u = torch.from_numpy(self._rng.random(int(sigma.shape[0])).astype(np.float32))
        u.mul_(1.0 - 2.0 * self.tail).add_(self.tail)
        z = torch.erfinv(u.mul(2.0).sub_(1.0)).mul_(_SQRT2)
        return mu.squeeze(-1) + sigma * z

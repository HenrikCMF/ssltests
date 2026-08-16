"""Exact (epsilon, delta) accounting for the client-level local DP defense.

WHAT IS BEING PROTECTED
-----------------------
The unit of privacy is ONE CLIENT'S ENTIRE DATASET, and the guarantee holds in
the LOCAL model: the client noises its own update before transmission, so the
guarantee is against the *server itself*. That is the only granularity that
means anything here — LOKI's adversary IS the aggregator, so central DP (noise
added after aggregation) would be adding noise on the attacker's honour.

Client-level DP is strictly stronger than example-level: if the server cannot
tell whether client k held dataset D or D', it certainly cannot reconstruct an
individual image out of D. So an epsilon reported here upper-bounds the
example-level epsilon that the gradient-inversion literature usually quotes.

WHY GDP AND NOT AN RDP ACCOUNTANT
---------------------------------
Every client participates in every round (fraction_fit=1.0), so there is no
subsampling amplification, and each round releases the SAME mechanism: a
Gaussian applied to an L2-clipped vector. For that setting Gaussian DP is
*exact* (Dong, Roth & Su 2022), whereas RDP would introduce conversion slack
for no reason. Composition is closed-form:

  * a Gaussian mechanism with L2 sensitivity D and per-coordinate noise std
    sigma is mu-GDP with mu = D / sigma                        (DRS Thm. 2.7)
  * T-fold composition of mu_i-GDP is sqrt(sum_i mu_i^2)-GDP   (DRS Cor. 3.3)
    => T identical rounds give mu_total = sqrt(T) * D / sigma

so the whole T-round transcript seen by the server is mu_total-GDP, and the
(eps, delta) curve of mu-GDP is closed-form (DRS Cor. 2.13):

  delta(eps) = Phi(mu/2 - eps/mu) - e^eps * Phi(-mu/2 - eps/mu)

Note this is identical to Balle & Wang's (2018) analytic Gaussian mechanism at
T = 1 — i.e. it is the tight Gaussian calibration, not the loose classical
sigma = D*sqrt(2 ln(1.25/delta))/eps, which is only valid for eps < 1 anyway
and would badly misreport the large-eps end of a sweep.

SENSITIVITY CONVENTION
----------------------
`SENSITIVITY_MULTIPLIER = 2` — neighbouring datasets are "client k's data
replaced wholesale". Both the real and the counterfactual update are clipped to
C, so ||u - u'|| <= 2C. An add/remove convention would give 1*C and therefore
roughly half the noise for the same epsilon; papers differ, so it is stated
explicitly rather than buried in a constant.
"""
from __future__ import annotations

import math

from scipy.special import log_ndtr

# Neighbouring relation: replace-one-client => L2 sensitivity is 2*C.
SENSITIVITY_MULTIPLIER = 2.0


def gdp_delta(eps: float, mu: float) -> float:
    """delta(eps) for a mu-GDP mechanism. Exact; monotone increasing in mu.

    The e^eps term is evaluated in log space: a sweep that reaches eps ~ 1e6
    (which this defense needs before the model trains at all) would overflow a
    naive math.exp(eps).
    """
    if mu <= 0.0:
        return 0.0
    t1 = math.exp(log_ndtr(mu / 2.0 - eps / mu))
    log_t2 = eps + log_ndtr(-mu / 2.0 - eps / mu)
    t2 = math.exp(log_t2) if log_t2 < 700.0 else math.inf
    return max(0.0, t1 - t2)


def _bisect(fn, target: float, lo: float, hi: float, increasing: bool,
            tol: float = 1e-12, iters: int = 200) -> float:
    """Bisect fn(x) == target on [lo, hi]. fn must be monotone."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        hit = fn(mid) < target
        if hit == increasing:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(1.0, lo):
            break
    return 0.5 * (lo + hi)


def _expand(fn, target: float, start: float, increasing: bool) -> float:
    """Grow a bracket until fn crosses target. Beats a hard-coded upper bound:
    this sweep reaches eps ~ 1e24, which needs mu ~ 1e12, and a fixed ceiling
    would silently return the cap instead of the answer."""
    hi = start
    for _ in range(200):
        crossed = (fn(hi) >= target) if increasing else (fn(hi) <= target)
        if crossed:
            return hi
        hi *= 2.0
    raise RuntimeError(f"could not bracket target={target}: reached {hi}")


def mu_for_eps_delta(eps: float, delta: float) -> float:
    """Largest mu whose (eps, delta) curve still satisfies the target."""
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if eps <= 0.0:
        raise ValueError(f"eps must be > 0, got {eps}")
    # delta is increasing in mu, from 0 (mu->0) to 1 (mu->inf).
    hi = _expand(lambda m: gdp_delta(eps, m), delta, 1.0, increasing=True)
    return _bisect(lambda m: gdp_delta(eps, m), delta, 1e-12, hi, increasing=True)


def noise_multiplier_for_eps(eps: float, delta: float, rounds: int) -> float:
    """z = sigma / sensitivity needed to hit (eps, delta) over `rounds` rounds."""
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")
    mu_total = mu_for_eps_delta(eps, delta)
    # mu_total = sqrt(T)/z  =>  z = sqrt(T)/mu_total
    return math.sqrt(rounds) / mu_total


def noise_std_for_eps(eps: float, delta: float, rounds: int, clip: float) -> float:
    """Absolute per-coordinate Gaussian std for a target (eps, delta)."""
    if clip <= 0.0:
        raise ValueError(f"clip must be > 0, got {clip}")
    sensitivity = SENSITIVITY_MULTIPLIER * clip
    return noise_multiplier_for_eps(eps, delta, rounds) * sensitivity


def eps_for_noise_std(sigma: float, delta: float, rounds: int, clip: float) -> float:
    """Inverse of `noise_std_for_eps`: the eps a given noise std actually buys.

    Places a run described by its absolute noise std on the eps axis — provided
    a clip norm is named for it, since without one there is no bounded
    sensitivity and hence no epsilon at all.
    """
    if sigma <= 0.0:
        return math.inf
    sensitivity = SENSITIVITY_MULTIPLIER * clip
    mu_total = math.sqrt(rounds) * sensitivity / sigma
    # delta is decreasing in eps at fixed mu.
    hi = _expand(lambda e: gdp_delta(e, mu_total), delta, 1.0, increasing=False)
    return _bisect(lambda e: gdp_delta(e, mu_total), delta, 1e-9, hi, increasing=False)


if __name__ == "__main__":
    # Self-checks: each states what would be wrong if it failed.
    # 1. At T=1 the formula must reproduce Balle & Wang's analytic Gaussian
    #    mechanism. Hand-computed: D=1, sigma=1 => mu=1; delta(eps=1) =
    #    Phi(-0.5) - e*Phi(-1.5).
    from scipy.stats import norm
    want = norm.cdf(-0.5) - math.e * norm.cdf(-1.5)
    got = gdp_delta(1.0, 1.0)
    assert abs(got - want) < 1e-12, f"GDP curve wrong at mu=1: {got} vs {want}"

    # 2. Round-trip: eps -> sigma -> eps must be the identity. The large end is
    #    load-bearing, not decoration: the LOKI sweep lives at eps ~ 1e16..1e24
    #    (a trap-dominated update norm makes the honest window that expensive),
    #    and a fixed bisection ceiling silently returns the cap instead of
    #    failing, which reads as a plausible sigma.
    for eps in (0.5, 1.0, 8.0, 1e3, 1e5, 1e16, 1e20, 1e24):
        s = noise_std_for_eps(eps, 1e-5, rounds=100, clip=1.0)
        back = eps_for_noise_std(s, 1e-5, rounds=100, clip=1.0)
        assert abs(back - eps) / eps < 1e-3, f"round-trip broke: {eps} -> {s} -> {back}"

    # 2b. sigma must be strictly decreasing in eps across that whole span.
    sigmas = [noise_std_for_eps(e, 1e-5, 100, 1e6)
              for e in (1e16, 1e18, 1e20, 1e22, 1e24)]
    assert all(a > b for a, b in zip(sigmas, sigmas[1:])), f"not monotone: {sigmas}"

    # 3. More rounds must cost privacy: sigma grows as sqrt(T).
    s1 = noise_std_for_eps(8.0, 1e-5, rounds=1, clip=1.0)
    s100 = noise_std_for_eps(8.0, 1e-5, rounds=100, clip=1.0)
    assert abs(s100 / s1 - 10.0) < 1e-6, f"sqrt(T) scaling broke: {s100 / s1}"

    print("dp_accounting: self-checks passed\n")
    print(f"{'eps':>10} {'mu_total':>10} {'z':>12} {'sigma/C':>12}")
    for eps in (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6):
        mu = mu_for_eps_delta(eps, 1e-5)
        z = noise_multiplier_for_eps(eps, 1e-5, 100)
        print(f"{eps:>10g} {mu:>10.4g} {z:>12.4g} {2.0 * z:>12.4g}")

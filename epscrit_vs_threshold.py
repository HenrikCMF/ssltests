"""eps_crit as a FUNCTION of the success threshold, not a single number.

Composes measured inputs only -- nothing derived, no fitted parameter:

  supply(eps, sigma*)  fragments per sample the attacker actually collects
                       = 2.7 clean fragments/sample/round x T_eff
                       T_eff = rounds whose fragment noise sigma_rec(t) =
                       sigma_DP/m_fired(t) still clears sigma*, using the measured
                       decay m_fired(t) ~= 3.5e-3/t and sigma_DP = 2C*z(eps) from
                       the repo's own accountant.

  m*(tau, sigma)       fragments NEEDED, measured by running the real inverter
                       (mstar_curve.csv + mstar_curve_variants.csv). Replaces the
                       van Trees derivation, and with it J_aug, the Gaussian image
                       prior, the known-A_i problem and the MSE non-monotonicity.

sigma* is the attacker's own keep/discard threshold, so eps_crit(tau) minimises
over it: the attacker trades fragment count against fragment quality and picks
the best point. That also removes E3 from the critical path -- the E3 route is
kept below only as a cross-check.

    python epscrit_vs_threshold.py
"""
from __future__ import annotations

import csv
import math

import numpy as np

from dp_accounting import noise_multiplier_for_eps

# ---- measured constants ----------------------------------------------------- #
C          = 1e6      # DP_CLIP; the clip never engaged, so this is the constant in sigma=2Cz
T          = 100      # NUM_ROUNDS (the accountant composes over this many)
T_ATTACK   = 99       # round 1 the trap is inert (server.py:348)
DELTA      = 1e-5
FRAGS_PER_ROUND = 2.7  # occupancy replay: ~27k fired bins / 10k samples, kick-out dynamics
LEAK_A     = 3.5e-3   # m_fired(t) ~= LEAK_A / t  (eps=1e26 log, rounds 2..50)
FRAG_RANGE = 1.377    # median (max-min) of a real fragment: display scale -> fragment scale

LP_BARS = [0.23, 0.3, 0.4, 0.5]
SS_BARS = [0.5, 0.6, 0.7]

# E3 cross-check only: vgg-LPIPS(corrupted view, true view) vs display-scale sigma.
E3_SIGMA = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80, 1.20]
E3_LPIPS = [0.009, 0.021, 0.053, 0.187, 0.376, 0.486, 0.551, 0.620, 0.676, 0.703, 0.714]


def sigma_star_e3(tau: float) -> float:
    return float(np.interp(tau, E3_LPIPS, E3_SIGMA)) * FRAG_RANGE


def dp_sigma(eps: float) -> float:
    return noise_multiplier_for_eps(eps, DELTA, T) * 2.0 * C


def t_eff(eps: float, sigma_star: float) -> float:
    if sigma_star <= 0:
        return 0.0
    return min(float(T_ATTACK), LEAK_A * sigma_star / dp_sigma(eps))


def supply(eps: float, sigma_star: float) -> float:
    return FRAGS_PER_ROUND * t_eff(eps, sigma_star)


def load_surface(paths=("mstar_curve.csv", "mstar_curve_variants.csv")):
    """{sigma: {m: {'lpips': [...], 'ssim': [...]}}} from every no-sign-flip row."""
    surf: dict = {}
    for p in paths:
        try:
            rows = list(csv.DictReader(open(p)))
        except FileNotFoundError:
            continue
        for r in rows:
            if r["sign_flip"] != "0":
                continue
            s, m = float(r["leak_noise"]), int(r["m"])
            surf.setdefault(s, {})[m] = {
                "lpips": [float(r[f"frac_lpips_lt_{b}"]) for b in LP_BARS],
                "ssim":  [float(r[f"frac_ssim_gt_{b}"]) for b in SS_BARS],
            }
    return surf


def _frac(curve, metric, tau):
    if metric == "lpips":
        return float(np.interp(tau, LP_BARS, curve["lpips"]))
    return float(np.interp(-tau, [-b for b in SS_BARS][::-1], curve["ssim"][::-1]))


def m_star(sig_rows, metric, tau) -> float:
    """Smallest m at which >=50% of samples clear the bar (log-interpolated)."""
    ms = sorted(sig_rows)
    fr = [_frac(sig_rows[m], metric, tau) for m in ms]
    for i, (m, f) in enumerate(zip(ms, fr)):
        if f >= 0.5:
            if i == 0:
                return float(m)
            m0, f0 = ms[i - 1], fr[i - 1]
            return math.exp(math.log(m0) + (0.5 - f0) / (f - f0) * (math.log(m) - math.log(m0)))
    return math.inf


def eps_crit_at(sigma_star, need) -> float:
    if not math.isfinite(need):
        return math.inf
    lo, hi = 1e14, 1e32
    if supply(hi, sigma_star) < need:
        return math.inf
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if supply(mid, sigma_star) >= need:
            hi = mid
        else:
            lo = mid
    return hi


if __name__ == "__main__":
    surf = load_surface()
    sigmas = sorted(surf)
    print(f"measured m*(tau, sigma) surface: sigma in {sigmas}, "
          f"m in {sorted(surf[sigmas[0]])}\n")

    print("1. m* MEASURED (fragments needed for >=50% of samples to clear the bar)")
    hdr = "  " + f"{'threshold':>14}" + "".join(f"{('s=' + str(s)):>9}" for s in sigmas)
    print(hdr)
    for metric, bars in (("lpips", LP_BARS[::-1]), ("ssim", SS_BARS)):
        for tau in bars:
            cells = []
            for s in sigmas:
                v = m_star(surf[s], metric, tau)
                cells.append(f"{v:>9.1f}" if math.isfinite(v) else f"{'>64':>9}")
            op = "<" if metric == "lpips" else ">"
            print(f"  {metric.upper() + ' ' + op + ' ' + str(tau):>14}" + "".join(cells))

    print("\n2. eps_crit(tau) -- minimised over the attacker's keep-threshold sigma*")
    print(f"  {'threshold':>14} {'best sigma*':>11} {'m* there':>9} {'eps_crit':>10}"
          f" {'T_eff':>7} {'supply':>8}   {'eps_crit (E3 route)':>20}")
    for metric, bars in (("lpips", LP_BARS[::-1]), ("ssim", SS_BARS)):
        for tau in bars:
            best = (math.inf, None, None)
            for s in sigmas:
                if s == 0.0:
                    continue                    # sigma*=0 collects nothing
                e = eps_crit_at(s, m_star(surf[s], metric, tau))
                if e < best[0]:
                    best = (e, s, m_star(surf[s], metric, tau))
            e, s, need = best
            op = "<" if metric == "lpips" else ">"
            lab = f"{metric.upper()} {op} {tau}"
            if math.isinf(e):
                print(f"  {lab:>14} {'-':>11} {'>64':>9} {'unreachable':>10}")
                continue
            e3 = ""
            if metric == "lpips":
                s3 = sigma_star_e3(tau)
                n3 = m_star(surf[min(sigmas, key=lambda x: abs(x - s3))], metric, tau)
                v3 = eps_crit_at(s3, n3)
                e3 = f"{v3:.2e}" if math.isfinite(v3) else "unreachable"
            print(f"  {lab:>14} {s:>11.2f} {need:>9.1f} {e:>10.2e} "
                  f"{t_eff(e, s):>7.1f} {supply(e, s):>8.1f}   {e3:>20}")

    print("\n3. Does demand bind? supply vs need at the sweep's own eps points")
    for tau in (0.3, 0.23):
        best_s = min((s for s in sigmas if s > 0),
                     key=lambda s: eps_crit_at(s, m_star(surf[s], "lpips", tau)))
        need = m_star(surf[best_s], "lpips", tau)
        print(f"  LPIPS < {tau} (attacker keeps sigma_rec <= {best_s}, needs m* = {need:.1f})")
        print(f"    {'eps':>8} {'sigma_DP':>10} {'T_eff':>7} {'supply':>8} {'supply/m*':>10} {'':>8}")
        for e in (1e20, 1e21, 1e22, 1e24, 1e26):
            sup = supply(e, best_s)
            print(f"    {e:>8.0e} {dp_sigma(e):>10.2e} {t_eff(e, best_s):>7.1f} {sup:>8.1f} "
                  f"{sup / need:>10.2f} {'RISK' if sup >= need else 'safe':>8}")

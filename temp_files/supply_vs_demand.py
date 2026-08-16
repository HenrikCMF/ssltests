"""
supply_vs_demand.py -- the model's prediction for every swept eps, under ONE gate.

Rebuilds Table 3 of privacy_model.pdf from its measured inputs so that the prediction
column and the observed column can be compared like-for-like:

    sigma_DP(eps) = 2C * sqrt(T / 2 eps)                          (Link 1, exact inversion
                                                                   from dp_accounting)
    T_eff(eps)    = min(T_leak, beta * sigma* / sigma_DP(eps))    (Link 3)
    supply(eps)   = f0 * T_eff(eps)                               (Link 4)
    m*(gate,sigma) from mstar_curve_variants.csv                  (Link 5, measured)

The one free parameter is the attacker's keep-threshold sigma*. It is NOT fitted to the
observed rates: it is chosen to MAXIMISE attacker strength, i.e. to maximise
supply/m* = f0*beta*sigma*/(sigma_DP * m*(sigma*)), which for an uncapped T_eff means
maximising sigma*/m*(sigma*) -- a quantity that depends only on the measured demand
surface and not on any observation. f0, beta and m* are each measured independently.

Everything else is read from artifacts: sigma_DP from dp_accounting (cross-checked against
dp_sweep_out/sweep.csv), m* from the inverter sweep, beta and f0 from the round logs.
"""
import csv
import json
import math
import os
import sys

from dp_accounting import noise_std_for_eps

# --- measured constants (privacy_model.pdf Table 1) ------------------------------- #
F0 = 2.7            # clean fragments per sample per round = nu*E*P_clean (Link 2, measured)
BETA = 3.5e-3       # leak-amplitude decay, m_fired(t) ~ beta/t (Link 3, measured)
T_LEAK = 99         # rounds carrying leakage (round 1 trap inert)
T_ROUNDS = 100      # rounds charged by DP composition
C_CLIP = 1e6        # DP_CLIP; never engages, and cancels out of sigma_rec anyway (§3.4)
DELTA = 1e-5

MSTAR_CSV = "mstar_curve_variants.csv"
# The two gates the implementation actually uses (correction ledger: "Gates are
# SSIM > 0.7 and vgg-LPIPS 0.23").
GATES = {"lpips<0.23": ("frac_lpips_lt_0.23", "gt"),
         "ssim>0.7": ("frac_ssim_gt_0.7", "gt")}


def load_mstar(path=MSTAR_CSV):
    """{gate: {sigma: m*}} -- smallest m at which >=50% of samples clear the gate.

    Log-interpolated between the measured m grid {1,2,4,...,64}, which is how Table 2
    was built (it reproduces 6.9 / 15.4 / 57.4 to the digit). Returns math.inf where the
    curve never reaches 50% within m <= 64."""
    rows = list(csv.DictReader(open(path)))
    by_sigma = {}
    for r in rows:
        by_sigma.setdefault(float(r["leak_noise"]), []).append(r)
    out = {g: {} for g in GATES}
    for g, (col, _) in GATES.items():
        for s, rs in by_sigma.items():
            rs = sorted(rs, key=lambda r: int(r["m"]))
            ms = [int(r["m"]) for r in rs]
            fr = [float(r[col]) for r in rs]
            m = math.inf
            for i in range(len(ms)):
                if fr[i] >= 0.5:
                    if i == 0:
                        m = float(ms[0])
                    else:
                        lo, hi = fr[i - 1], fr[i]
                        t = (0.5 - lo) / (hi - lo) if hi > lo else 1.0
                        m = 2 ** (math.log2(ms[i - 1]) + t * (math.log2(ms[i]) - math.log2(ms[i - 1])))
                    break
            out[g][s] = m
    return out


def best_sigma_star(mstar_gate):
    """sigma* maximising attacker strength, chosen WITHOUT reference to any observation.

    supply/m* = f0*beta*sigma* / (sigma_DP * m*(sigma*)) while T_eff is below its cap, so
    the attacker's optimum maximises sigma*/m*(sigma*) -- a property of the measured demand
    surface alone. sigma = 0 is excluded: it is the out-of-distribution slice for the
    deployed checkpoint (Remark 4) and is never the attacker's choice."""
    cands = [(s, m) for s, m in mstar_gate.items() if s > 0 and math.isfinite(m)]
    if not cands:
        return None, math.inf
    s, m = max(cands, key=lambda sm: sm[0] / sm[1])
    return s, m


def predict(eps, sigma_star, mstar):
    sigma_dp = noise_std_for_eps(eps, DELTA, rounds=T_ROUNDS, clip=C_CLIP)
    t_eff_raw = BETA * sigma_star / sigma_dp
    t_eff = min(T_LEAK, t_eff_raw)
    supply = F0 * t_eff
    return {"eps": eps, "sigma_dp": sigma_dp, "t_eff_uncapped": t_eff_raw,
            "t_eff": t_eff, "supply": supply, "mstar": mstar,
            "supply_over_mstar": supply / mstar}


def eps_crit(sigma_star, mstar):
    """supply = m*  =>  eps_crit = 2 T C^2 (m*/(beta f0 sigma*))^2   (Eq. 6)."""
    return 2 * T_ROUNDS * C_CLIP ** 2 * (mstar / (BETA * F0 * sigma_star)) ** 2


def main():
    eps_grid = [float(x) for x in (sys.argv[1:] or
                ["1e2", "1e18", "1e19", "1e20", "1e21", "1e22", "1e24", "1e26"])]
    ms = load_mstar()
    out = {}
    for gate in GATES:
        s, m = best_sigma_star(ms[gate])
        print(f"\n=== gate {gate} ===")
        print("  measured demand surface m*(sigma): " +
              "  ".join(f"{k}:{'>64' if math.isinf(v) else f'{v:.1f}'}"
                        for k, v in sorted(ms[gate].items())))
        print(f"  attacker-optimal sigma* = {s} -> m* = {m:.1f}   "
              f"(maximises sigma*/m*, not fitted to observations)")
        print(f"  eps_crit = {eps_crit(s, m):.2g}")
        rows = [predict(e, s, m) for e in eps_grid]
        print(f"\n  {'eps':>8} {'sigma_DP':>10} {'T_eff':>8} {'supply':>8} {'supply/m*':>10}  regime")
        for r in rows:
            reg = ("safe" if r["supply_over_mstar"] < 0.5 else
                   "marginal" if r["supply_over_mstar"] < 2 else "at risk")
            cap = "*" if r["t_eff_uncapped"] > T_LEAK else " "
            print(f"  {r['eps']:>8.0e} {r['sigma_dp']:>10.3g} {r['t_eff']:>7.2f}{cap} "
                  f"{r['supply']:>8.1f} {r['supply_over_mstar']:>10.3f}  {reg}")
        out[gate] = {"sigma_star": s, "mstar": m, "eps_crit": eps_crit(s, m),
                     "rows": rows}
    json.dump(out, open("predictions.json", "w"), indent=2, default=str)
    print("\nwrote predictions.json   (* = T_eff capped at T_leak = 99)")


if __name__ == "__main__":
    main()

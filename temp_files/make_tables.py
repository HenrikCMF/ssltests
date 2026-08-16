"""
make_tables.py -- join predictions.json (model) with results_arms.json (measured) and
print the markdown tables for privacy_findings.md.

Nothing here computes a metric; it only formats. The two inputs are produced by
supply_vs_demand.py (model side, no observation used) and score_recons.py (measured side,
one fixed gate for every arm).
"""
import json
import sys

EPS_ARMS = [("1e+02", 1e2, "full02"), ("1e+18", 1e18, "full18"), ("1e+20", 1e20, "full20"),
            ("1e+22", 1e22, "full22"), ("1e+24", 1e24, "full24"), ("1e+26", 1e26, "full26")]

TRUNC_ARMS = [
    ("A  full 1e22, all 99 rounds", "full22"),
    ("C  real 1e22, rounds 2-7 only", "trunc22_r2_7"),
    ("B  1e26 rounds 2-7 + matched noise", "emul22_r2_7"),
    ("D  1e26 rounds 2-7, no noise", "clean26_r2_7"),
    ("E  full 1e26, all 99 rounds", "full26"),
]


def pct(r, key):
    return "—" if r is None else f"{r.get(key, 0.0):.1f}%"


def main():
    pred = json.load(open("predictions.json"))
    res = json.load(open("results_arms.json"))
    gate = sys.argv[1] if len(sys.argv) > 1 else "lpips<0.23"
    gcol = {"lpips<0.23": "distinct_lpips_lt_0.23_pct",
            "ssim>0.7": "distinct_ssim_gt_0.7_pct"}[gate]

    print(f"### Sweep under one fixed criterion — gate {gate}, "
          f"sigma* = {pred[gate]['sigma_star']} (m* = {pred[gate]['mstar']:.1f})\n")
    print("| ε | σ_DP | T_eff | supply | supply/m\\* | predicted | observed (distinct "
          "client-0 leaked) | bins | nearest-match |")
    print("|---|---|---|---|---|---|---|---|---|")
    prows = {f"{float(r['eps']):.0e}": r for r in pred[gate]["rows"]}
    for tag, eps, arm in EPS_ARMS:
        p = prows.get(f"{eps:.0e}")
        r = res.get(arm)
        reg = "—"
        if p:
            s = float(p["supply_over_mstar"])
            reg = "safe" if s < 0.5 else ("**marginal**" if s < 2 else "at risk")
        print(f"| {tag} | {float(p['sigma_dp']):.2g} | {float(p['t_eff']):.2f} | "
              f"{float(p['supply']):.1f} | **{float(p['supply_over_mstar']):.3f}** | {reg} | "
              f"{pct(r, gcol)} | {0 if r is None else r.get('n_bins_reconstructed', 0)} | "
              f"{pct(r, 'distinct_leaked_nearest_pct')} |")

    print("\n### Round-truncation test\n")
    print("| arm | bins (≥8 frags) | nearest-match | @SSIM>0.7 | @LPIPS<0.23 |")
    print("|---|---|---|---|---|")
    for label, arm in TRUNC_ARMS:
        r = res.get(arm)
        print(f"| {label} | {0 if r is None else r.get('n_bins_reconstructed', 0)} | "
              f"{pct(r, 'distinct_leaked_nearest_pct')} | {pct(r, 'distinct_ssim_gt_0.7_pct')} | "
              f"{pct(r, 'distinct_lpips_lt_0.23_pct')} |")

    print("\n### Sensitivity of the truncation arms\n")
    print("| arm | K_NN=99, MIN_VIEWS=8 | K_NN=6, MIN_VIEWS=8 | K_NN=99, MIN_VIEWS=4 |")
    print("|---|---|---|---|")
    for base in ["clean26_r2_7", "trunc22_r2_7", "emul22_r2_7"]:
        cells = []
        for suf in ["", "_k6", "_mv4"]:
            r = res.get(base + suf)
            cells.append(f"{pct(r, 'distinct_leaked_nearest_pct')} nearest / "
                         f"{pct(r, 'distinct_ssim_gt_0.7_pct')} @SSIM>0.7")
        print(f"| {base} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()

"""
check_emulation.py -- does the synthetic sigma_rec(t) actually reproduce the real
eps=1e22 fragment noise, per fragment rather than on average?

The round-cutoff test replaces real eps=1e22 fragments with eps=1e26 fragments plus
Gaussian noise of std sigma_rec(t) = sigma_DP / m_fired_median(t). That uses ONE sigma per
round, but m_fired spans ~100x across bins in every logged round (min/median/max), so real
dim bins are far noisier than the median and real bright bins far cleaner. This script
measures the realized per-fragment noise in all three sets and compares the distributions,
so the emulation's error is quantified instead of assumed.

Estimator: a fragment is a crop of a natural image, so adjacent pixels are strongly
correlated and the horizontal first difference is dominated by the additive noise. For
i.i.d. noise of std s the difference has std s*sqrt(2), so the robust estimate is

    s_hat = MAD(dx) / 0.6745 / sqrt(2)

MAD rather than std so that a handful of genuine edges cannot inflate it. The estimate is
made in the saved fragment's own max-abs-1 space, which is the space sigma_rec is defined in.
"""
import os

import torch

SETS = {
    "clean 1e26": "fragments_eps_1e+26",
    "real 1e22": "fragments_eps_1e+22",
    "emul 1e22": "frag_emul22_r2_7",
}
ROUNDS = range(2, 8)


def noise_est(f):
    """Robust per-fragment noise std from the horizontal first difference. f [n,3,32,32]."""
    dx = (f[..., 1:] - f[..., :-1]).flatten(1)
    mad = (dx - dx.median(dim=1, keepdim=True).values).abs().median(dim=1).values
    return mad / 0.6745 / (2 ** 0.5)


def main():
    print(f"{'round':>5} " + " ".join(f"{k:>34}" for k in SETS))
    print(f"{'':>5} " + " ".join(f"{'p10     median      p90':>34}" for _ in SETS))
    stats = {k: {} for k in SETS}
    for t in ROUNDS:
        cells = []
        for k, d in SETS.items():
            p = os.path.join(d, f"round_{t:03d}.pt")
            f = torch.load(p, map_location="cpu", weights_only=False)["frags"].float()
            s = noise_est(f)
            q = [float(s.quantile(q_)) for q_ in (0.1, 0.5, 0.9)]
            stats[k][t] = q
            cells.append(f"{q[0]:>10.4f} {q[1]:>10.4f} {q[2]:>10.4f}   ")
        print(f"{t:>5} " + " ".join(cells), flush=True)

    print("\nrealized noise, emulation vs real (median-bin estimate), and the spread each carries:")
    print(f"{'round':>5} {'real med':>10} {'emul med':>10} {'ratio':>8} "
          f"{'real p90/p10':>13} {'emul p90/p10':>13}")
    for t in ROUNDS:
        r, e = stats["real 1e22"][t], stats["emul 1e22"][t]
        print(f"{t:>5} {r[1]:>10.4f} {e[1]:>10.4f} {e[1] / max(r[1], 1e-9):>8.2f} "
              f"{r[2] / max(r[0], 1e-9):>13.2f} {e[2] / max(e[0], 1e-9):>13.2f}")


if __name__ == "__main__":
    main()

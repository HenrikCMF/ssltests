"""
make_truncation_arms.py -- build the fragment sets for the round-cutoff test (§3.2 of
privacy_model.pdf, "Test the round-cutoff directly").

Link 3 of the model claims the DP noise imposes a *hard cutoff round* rather than a
smooth degradation:

    sigma_rec(t) = sigma_DP / m_fired(t),   T_eff = min(T_leak, beta*sigma* / sigma_DP)

so at eps=1e22 only rounds 2..6-7 carry fragments an attacker would keep, and rounds
8..100 are worthless. That claim is what makes this a cliff model (eps_crit ~ (m*/sigma*)^2)
and it has never been tested directly. This script builds the fragment directories for
four arms; run_arm.sh then puts each through the unmodified regroup -> infer -> score
pipeline.

  A  full22     fragments_eps_1e+22            all 99 rounds        (reference: the real run)
  C  trunc22    rounds 2..7 of the real 1e22   attacker discards 8+ (prediction: == A)
  B  emul22     rounds 2..7 of 1e26 + noise    synthetic sigma_rec  (prediction: == A)
  D  clean26_7  rounds 2..7 of 1e26, no noise  noise-free control   (isolates noise from rounds)

Why the emulation is a faithful stand-in
----------------------------------------
A saved fragment is Eq. 9: row_i / max|row_i|, i.e. the clean signal shape with max-abs 1.
At eps=1e26 sigma_DP = 1.41e-6 against m_fired(t) ~ 1.5e-3..5.5e-4, so sigma_rec <= 0.003:
those fragments are the clean shape f_t. The real eps=1e22 fragment is

    (m_fired(t) * f_t + n) / max|m_fired(t) * f_t + n|,  n ~ N(0, sigma_DP^2) per coord
  = (f_t + sigma_rec(t) * g) / max|f_t + sigma_rec(t) * g|,  g ~ N(0,1)

which is exactly what this script constructs -- add sigma_rec(t)*randn in the max-abs-1
fragment space, then renormalize by max-abs. This is the same space and the same
convention reconstruction_test.leak_views uses for LEAK_NOISE (noise injected pre-to_unit,
where it does not cancel).

sigma_rec(t) uses the MEASURED m_fired median of the eps=1e26 run (dp_sweep_out/eps_1e+26.log),
not the beta/t fit -- the fit is within ~20% of it and the measurement is strictly better
data. The noise-model constant it rests on is validated to 3 digits: logged m_fired at the
round-100 floor is 3.688-3.690 x sigma_DP at every one of the five eps points.

KNOWN APPROXIMATION: m_fired spans ~100x across bins (log records min/median/max per round),
so a single scalar sigma_rec(t) gives every fragment the MEDIAN bin's noise; in the real run
dim bins are noisier and bright bins cleaner. That is precisely what arm C (real 1e22
fragments, same rounds) controls for -- the B-vs-C gap bounds the cost of this approximation.
"""
import os
import re
import shutil
import sys

import torch

SRC22 = "fragments_eps_1e+22"
SRC26 = "fragments_eps_1e+26"
LOG26 = "dp_sweep_out/eps_1e+26.log"

SIGMA_DP = {"1e+22": 1.4142135624155652e-4,   # dp_sweep_out/sweep.csv
            "1e+26": 1.4142135623737901e-6}

ROUNDS = tuple(range(2, 8))          # rounds 2..7, the keepable window at eps=1e22
SEED = 20260727

_LOKI = re.compile(r"r(\d+) loki\(cid0\): fc1w_rms=\S+\s+m_fired median=([\d.e+-]+)")


def m_fired_profile(log_path):
    """round -> median m_fired over fired bins, as logged by the server each round."""
    out = {}
    with open(log_path) as fh:
        for line in fh:
            m = _LOKI.search(line)
            if m:
                out[int(m.group(1))] = float(m.group(2))
    return out


def link_arm(dst, src, rounds):
    """Arm made of unmodified round files -> symlink (no 3 GB copy per arm)."""
    os.makedirs(dst, exist_ok=True)
    for t in rounds:
        name = f"round_{t:03d}.pt"
        s, d = os.path.abspath(os.path.join(src, name)), os.path.join(dst, name)
        if os.path.lexists(d):
            os.remove(d)
        os.symlink(s, d)
    print(f"{dst}: {len(rounds)} rounds symlinked from {src}")


def noise_arm(dst, src, rounds, sigma_rec, seed=SEED):
    """Arm made of eps=1e26 fragments carrying the sigma_rec(t) a 1e22 run would have had.

    Reproduces the real chain exactly: perturb in max-abs-1 space, then re-apply Eq. 9's
    max-abs normalizer to the *noisy* row (which is what inflates the normalizer once the
    noise approaches the signal peak -- at sigma_rec = 0.26 the max of 3072 Gaussians is
    ~0.95, comparable to the signal's own peak of 1)."""
    os.makedirs(dst, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    for t in rounds:
        name = f"round_{t:03d}.pt"
        obj = torch.load(os.path.join(src, name), map_location="cpu", weights_only=False)
        f = obj["frags"].float()                                   # [n,3,32,32], max-abs 1
        s = sigma_rec[t]
        noisy = f + s * torch.randn(f.shape, generator=g)
        m = noisy.flatten(1).abs().amax(1).clamp_min(1e-12).view(-1, 1, 1, 1)
        obj["frags"] = (noisy / m).float()
        torch.save(obj, os.path.join(dst, name))
        print(f"  {name}: n={f.shape[0]} sigma_rec={s:.4f} "
              f"(max-abs pre-renorm median {float(noisy.flatten(1).abs().amax(1).median()):.3f})",
              flush=True)
    print(f"{dst}: {len(rounds)} rounds written from {src} + matched noise")


def main():
    mf26 = m_fired_profile(LOG26)
    sigma_rec = {t: SIGMA_DP["1e+22"] / mf26[t] for t in ROUNDS}
    print("sigma_rec(t) an eps=1e22 run would have had (sigma_DP / measured m_fired median):")
    for t in ROUNDS:
        keep = "keep" if sigma_rec[t] <= 0.25 else "DISCARD (> sigma*=0.25)"
        print(f"  r{t}: m_fired={mf26[t]:.4g}  sigma_rec={sigma_rec[t]:.4f}   {keep}")
    beta = 3.5e-3
    print("model check, sigma_rec = sigma_DP*t/beta with beta=3.5e-3: " +
          " ".join(f"r{t}:{SIGMA_DP['1e+22'] * t / beta:.3f}" for t in ROUNDS))

    if "--dry-run" in sys.argv:
        return
    link_arm("frag_trunc22_r2_7", SRC22, ROUNDS)
    link_arm("frag_clean26_r2_7", SRC26, ROUNDS)
    noise_arm("frag_emul22_r2_7", SRC26, ROUNDS, sigma_rec)

    # Round partition of the eps=1e22 run: 2-7 | 8-20 | 21-100, disjoint and exhaustive.
    # Needed because "rounds 2-7 alone leak 5x less than all 99 rounds" has two readings:
    # (a) rounds 8+ carry real signal, so T_eff's hard cutoff is wrong; or (b) rounds 8+
    # are pure noise but their fragments still supply the neighbourhood DENSITY that lets
    # DBSCAN find the good early-round fragments (MIN_PTS = 4 over a kNN graph), in which
    # case supply is not what the extra rounds are contributing. Running the late blocks
    # ALONE separates the two: under (b) they must leak ~nothing on their own.
    link_arm("frag_late22_r8_20", SRC22, range(8, 21))
    link_arm("frag_late22_r21_100", SRC22, range(21, 101))
    # NULL ARM. At rounds 91-100 the eps=1e22 run's logged m_fired median is 5.218e-4 =
    # 3.69 * sigma_DP, i.e. exactly the max-abs-of-3072-Gaussians noise floor (Remark 1):
    # by direct measurement these rounds contain NO signal. So whatever this arm scores is
    # the floor of the scoring procedure itself -- clusters of pure noise still get a
    # nearest neighbour in a 50k pool, and some of those land in client 0 by chance. Every
    # other arm must be read against this number, not against zero.
    link_arm("frag_null22_r91_100", SRC22, range(91, 101))


if __name__ == "__main__":
    main()

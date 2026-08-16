#!/usr/bin/env python
"""Build the per-C fragment showcases and the knn results table for the clip sweep.

For every arm of dp_clip_sweep.py, writes two 10x10 grids of RANDOMLY sampled
LOKI fragments -- one from round 2 (earliest round that produces fragments) and
one from round 10 (the last executed round) -- plus RESULTS.md with the round-10
knn accuracy per C.

Fragments are mapped to viewable images with the canonical per-image min-max of
regroup_fragments.to_unit, which is the same map the clustering pipeline applies.
That matters: a fragment's absolute scale carries no information (LOKI reads it
out up to a per-bin scale), so min-max is the honest rendering. It also means a
pure-noise fragment renders as full-contrast noise rather than a flat grey field
-- the grids show STRUCTURE, not amplitude.

USAGE
    python fragment_showcase.py            # all arms found on disk
    python fragment_showcase.py 1e+02      # one arm
"""
from __future__ import annotations

import csv
import glob
import os
import sys
from pathlib import Path

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dp_accounting import noise_std_for_eps
from dp_clip_sweep import (CLIP_GRID, DP_DELTA, DP_EPS, EPS_TAG, NUM_ROUNDS,
                           OUT, SIM_ROUNDS, frag_dest, model_dest, parse, tag_of)

GRID = 10                     # 10x10 = 100 fragments per showcase
SHOWCASE_ROUNDS = (2, SIM_ROUNDS)
SHOW_DIR = OUT / "showcases"
SEED = 12345                  # same sample of bins across arms where possible


def to_unit(x: torch.Tensor) -> torch.Tensor:
    """Per-image min-max to [0,1] -- the canonical fragment->image map.

    Duplicated from regroup_fragments rather than imported: that module runs a
    heavy CUDA/model import chain at module scope, and this script only needs the
    four lines below.
    """
    lo = x.amin(dim=(-3, -2, -1), keepdim=True)
    hi = x.amax(dim=(-3, -2, -1), keepdim=True)
    return (x - lo) / (hi - lo + 1e-8)


def load_round(frag_dir: Path, rnd: int):
    """Return the [n,C,H,W] fragment stack for one round, or None if absent."""
    p = frag_dir / f"round_{rnd:03d}.pt"
    if not p.exists():
        return None
    payload = torch.load(p, map_location="cpu")
    frags = payload["frags"] if isinstance(payload, dict) else payload
    return frags.float()


def showcase(frags: torch.Tensor, path: Path, title: str) -> dict:
    """Save a GRID x GRID panel of randomly sampled fragments. Returns their stats."""
    n = frags.shape[0]
    want = GRID * GRID
    g = torch.Generator().manual_seed(SEED)
    # Sample WITHOUT replacement when there are enough fragments; when the round
    # produced fewer than 100, show them all and leave the rest of the panel blank
    # rather than duplicating (a duplicated grid would misread as "100 leaks").
    idx = torch.randperm(n, generator=g)[:want]
    sel = frags[idx]
    imgs = to_unit(sel)

    fig, axes = plt.subplots(GRID, GRID, figsize=(GRID * 1.1, GRID * 1.15))
    for r in range(GRID):
        for c in range(GRID):
            ax = axes[r][c]
            ax.axis("off")
            k = r * GRID + c
            if k < len(imgs):
                ax.imshow(imgs[k].permute(1, 2, 0).numpy())
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    # Per-fragment dynamic range, pre-normalization: the number that separates a
    # real readout from noise once min-max has hidden the amplitude.
    rng = (sel.amax(dim=(-3, -2, -1)) - sel.amin(dim=(-3, -2, -1)))
    return {"n_total": n, "n_shown": len(imgs),
            "range_median": float(rng.median()), "range_max": float(rng.max())}


def param_count() -> int | None:
    """Number of float params the DP vector spans, from any archived eval model.

    Read off a real checkpoint rather than hardcoded, so the SNR figure below can
    never silently drift out of date if the architecture changes. Returns None if
    no arm has finished yet (the section is then simply omitted).
    """
    for c in CLIP_GRID:
        p = model_dest(c)
        if not p.exists():
            continue
        try:
            sd = torch.load(p, map_location="cpu")
            return sum(v.numel() for v in sd.values()
                       if torch.is_tensor(v) and v.is_floating_point())
        except Exception as e:
            print(f"[showcase] could not read {p} for param count: {e}")
    return None


def results_by_tag() -> dict:
    """knn results keyed by clip tag, re-parsed from each arm's LOG.

    The logs are re-parsed rather than read from clip_sweep.csv so this script is
    authoritative: the CSV is written by whatever version of the driver happened to
    be running, and re-parsing also recovers arms from a sweep that died partway
    (the CSV is only rewritten between arms). Falls back to the CSV for any arm
    whose log is missing.
    """
    out = {}
    for c in CLIP_GRID:
        t = tag_of(c)
        log = OUT / f"clip_{t}.log"
        if log.exists():
            out[t] = parse(log.read_text(errors="replace"))
    p = OUT / "clip_sweep.csv"
    if p.exists():
        with open(p) as f:
            for r in csv.DictReader(f):
                out.setdefault(tag_of(float(r["clip"])), r)
    return out


def _val(row: dict, key: str):
    """Read a field that may be a float (log-parsed) or a string (CSV-loaded)."""
    v = row.get(key)
    if v in (None, "", "None"):
        return None
    return float(v)


def tv_score(frags: torch.Tensor) -> torch.Tensor:
    """Mean abs. neighbour difference per fragment, after the min-max map.

    Separates a real image readout from noise WITHOUT needing ground truth: a LOKI
    fragment of a natural image is spatially smooth (low TV), Gaussian noise is not
    (high TV). Scale-invariant, because to_unit has already removed the amplitude —
    which is the point, since the clip only ever changes amplitude.
    """
    u = to_unit(frags)
    dh = (u[..., :, 1:] - u[..., :, :-1]).abs().mean(dim=(-3, -2, -1))
    dv = (u[..., 1:, :] - u[..., :-1, :]).abs().mean(dim=(-3, -2, -1))
    return 0.5 * (dh + dv)


def noise_threshold():
    """TV cutoff calibrated on fragments KNOWN to be pure noise.

    The null is an eps=3e2 arm: there the model was annihilated and every panel is
    visually noise, so its TV distribution is the empirical null. Taking its 1st
    percentile fixes the false-positive rate at 1% by construction, so a leak rate
    far above 1% cannot be an artefact of the threshold. Returns None if the eps=3e2
    sweep is not on disk, in which case the leak column is simply omitted rather
    than being computed against a guessed constant.
    """
    for c in (1.0, 10.0, 100.0, 1000.0):
        d = Path(f"fragments_clip_eps3e+02_C{tag_of(c)}")
        f = load_round(d, SIM_ROUNDS) if d.is_dir() else None
        if f is not None:
            return float(tv_score(f).quantile(0.01)), f"eps=3e2 C={tag_of(c)} r{SIM_ROUNDS}"
    return None, None


def fmt_knn(row: dict) -> str:
    if not row:
        return "not run"
    if str(row.get("knn_r10_nan")) == "True":
        return "**nan** (diverged)"
    v = _val(row, "knn_r10")
    return "no r10 eval" if v is None else f"{v:.2f}%"


def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    tags = [tag_of(c) for c in CLIP_GRID]
    if only:
        tags = [t for t in tags if t == only] or [only]

    res = results_by_tag()
    SHOW_DIR.mkdir(parents=True, exist_ok=True)

    # ---- pass 1: render showcases and measure the leak ----------------------
    # Done before the table is built because the leak rate is a table column. Each
    # round's fragment stack is ~490 MB, so it is loaded exactly once here.
    thr, null_desc = noise_threshold()
    leak: dict = {}
    show_lines = ["", "## Fragment showcases", "",
                  f"{GRID}x{GRID} randomly sampled fragments per panel (seed {SEED}), "
                  "rendered with the canonical per-image min-max map. Absolute scale "
                  "is not shown by design, so a noise-only fragment appears as "
                  "full-contrast noise, not grey.", ""]
    for c in CLIP_GRID:
        t = tag_of(c)
        if t not in tags:
            continue
        frag_dir = frag_dest(c)
        if not frag_dir.is_dir():
            print(f"[showcase] C={t}: {frag_dir} missing, skipping")
            show_lines += [f"### C = {c:.0e}", "", "_arm not run_", ""]
            continue
        show_lines += [f"### C = {c:.0e}", ""]
        for rnd in SHOWCASE_ROUNDS:
            frags = load_round(frag_dir, rnd)
            if frags is None:
                avail = sorted(os.path.basename(p) for p in
                               glob.glob(str(frag_dir / "round_*.pt")))
                print(f"[showcase] C={t} r{rnd}: no file (have {avail})")
                show_lines += [f"- round {rnd}: _no fragments file_", ""]
                continue
            out = SHOW_DIR / f"clip_{t}_round{rnd:03d}.png"
            title = (f"LOKI fragments — C={c:.0e}, eps={DP_EPS:g}, round {rnd}  "
                     f"(sigma={noise_std_for_eps(DP_EPS, DP_DELTA, NUM_ROUNDS, c):.4g})")
            st = showcase(frags, out, title)
            lr = None
            if thr is not None:
                lr = float((tv_score(frags) < thr).float().mean()) * 100.0
                leak.setdefault(t, {})[rnd] = lr
            print(f"[showcase] C={t} r{rnd}: {st['n_shown']}/{st['n_total']} "
                  f"fragments, leak={'n/a' if lr is None else f'{lr:.1f}%'} -> {out}")
            show_lines += [
                f"- **round {rnd}** — {st['n_total']} fragments this round"
                + (f", **{lr:.1f}% structured** (vs 1% on the noise null)"
                   if lr is not None else "")
                + f", median dynamic range {st['range_median']:.4g}",
                "",
                f"  ![C={c:.0e} round {rnd}]({out.relative_to(OUT).as_posix()})",
                "",
            ]

    # ---- pass 2: header and table ------------------------------------------
    # The untrained reference, per arm. NOT bit-identical across arms (build_models
    # is not seeded, so each run's random init differs slightly — observed spread is
    # ~0.1 pt), which is why `gain` below is always computed against the arm's OWN
    # r0 rather than a single sweep-wide constant. The header quotes the range so
    # that spread is visible instead of being silently averaged away.
    bases = [_val(r, "knn_r0") for r in res.values() if _val(r, "knn_r0") is not None]
    if not bases:
        base_s = "n/a"
    elif max(bases) - min(bases) < 0.005:
        base_s = f"{bases[0]:.2f}%"
    else:
        base_s = f"{min(bases):.2f}–{max(bases):.2f}% (n={len(bases)} arms)"
    lines = [
        "# DP clip sweep — fragment showcases and utility",
        "",
        f"Fixed budget: **eps = {DP_EPS:g}**, delta = {DP_DELTA:g}, nominal horizon "
        f"{NUM_ROUNDS} rounds (sqrt(T) composition), **{SIM_ROUNDS} rounds executed** "
        "per arm. Only `C` varies.",
        "",
        "`sigma` is calibrated FROM `C` (`dp_accounting.noise_std_for_eps`: "
        "sigma = 2*C*z), so `sigma/C` is constant across the grid by construction — "
        "that constancy is the thing under test. `clip_factor` is the measured "
        "`min(1, C/||u||)` from the last round's client-0 log line: below 1 means the "
        "clip actually bound the update.",
        "",
        f"**Reference line: kNN of the UNTRAINED model (round 0) = {base_s}.** "
        "That is random-init BYOL features, not the 10% class prior (10%). Read "
        "every arm against it: above = BYOL learned, equal = the update never moved "
        "the model, below = the noise destroyed the initial features. `gain` is "
        f"`knn@r{SIM_ROUNDS}` minus **that arm's own** r0.",
        "",
        "The **leak** column is the fraction of that round's fragments whose spatial "
        "total variation falls below the 1st percentile of a KNOWN-noise null"
        + (f" ({null_desc})" if null_desc else "") +
        ". A natural-image readout is smooth, Gaussian noise is not; the map is "
        "applied after per-image min-max, so it measures STRUCTURE and is blind to "
        "amplitude — exactly the property the clip changes. The null fixes the "
        "false-positive rate at 1%, so any value far above 1% is a real leak.",
        "",
        "| C | sigma | sigma/C | clip_factor | clip active | knn @ r0 | knn @ r5 | "
        f"knn @ r{SIM_ROUNDS} | gain vs init | leak r2 | leak r{SIM_ROUNDS} |",
        "|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in CLIP_GRID:
        t = tag_of(c)
        row = res.get(t, {})
        sigma = noise_std_for_eps(DP_EPS, DP_DELTA, NUM_ROUNDS, c)
        cf = _val(row, "clip_factor")
        cf_s = f"{cf:.3g}" if cf is not None else "—"
        act_s = {"True": "yes", "False": "**no**"}.get(str(row.get("clip_active")), "—")
        k0 = _val(row, "knn_r0")
        k0_s = f"{k0:.2f}%" if k0 is not None else "—"
        k5 = _val(row, "knn_r5")
        k5_s = f"{k5:.2f}%" if k5 is not None else "—"
        g = _val(row, "gain_vs_init")
        g_s = f"{g:+.2f} pt" if g is not None else "—"
        lk = leak.get(t, {})
        l2 = f"{lk[2]:.1f}%" if 2 in lk else "—"
        l10 = f"{lk[SIM_ROUNDS]:.1f}%" if SIM_ROUNDS in lk else "—"
        lines.append(f"| {c:.0e} | {sigma:.4g} | {sigma / c:.4g} | {cf_s} | {act_s} "
                     f"| {k0_s} | {k5_s} | {fmt_knn(row)} | {g_s} | {l2} | {l10} |")

    # Why every arm lands in the same place, stated as a number rather than a claim.
    # Post-clip the signal has norm exactly C; the noise has norm sigma*sqrt(d) =
    # (2z*C)*sqrt(d). C cancels, so the SNR of the transmitted update depends only on
    # eps (through z) and the model size — never on the clip.
    d = param_count()
    if d:
        z2 = noise_std_for_eps(DP_EPS, DP_DELTA, NUM_ROUNDS, 1.0)   # = sigma/C
        snr = 1.0 / (z2 * (d ** 0.5))
        lines += [
            "",
            "## Why C cannot matter (the quantity the sweep is testing)",
            "",
            "Once the clip is active the transmitted update is exactly",
            "",
            "```",
            "    clip(u, C) + N(0, sigma^2)  =  C * [ u/||u|| + N(0, (sigma/C)^2) ]",
            "```",
            "",
            f"so its signal norm is `C` and its noise norm is "
            f"`(sigma/C)*sqrt(d)*C`. **C cancels.** With "
            f"`d = {d:,} float parameters` and `sigma/C = {z2:.4g}`:",
            "",
            f"    SNR = 1 / ((sigma/C) * sqrt(d)) = {snr:.3g}",
            "",
            f"i.e. the honest signal is about **1/{1 / snr:,.0f} of the noise it is "
            "buried in**, identically at every C in the grid. C rescales signal and "
            "noise together, so the optimizer sees only a learning-rate change — "
            "never a better-conditioned update. (`d` counts the encoder's float "
            "params, which carry the LOKI trap and dominate; the predictor adds "
            "well under 1% and does not move the figure.)",
        ]

    # Arms with no log at all were never started (the sweep was stopped, or has not
    # reached them yet). Say so explicitly: a bare "not run" row otherwise reads as
    # a crashed arm, and if C=1e6 is among them the sweep has no clip-INACTIVE
    # control, which bounds what the whole table is allowed to claim.
    missing = [tag_of(c) for c in CLIP_GRID if tag_of(c) not in res]
    if missing:
        note = (f"> **Arms not run: {', '.join(missing)}.** These were never "
                "started, not failed — the rows are marked *not run* below.")
        if tag_of(1e6) in missing:
            note += (" C = 1e6 is the only grid point above ||u||, so without it "
                     "this sweep has **no clip-INACTIVE control** and its conclusion "
                     "stands for the clip-active regime only.")
        lines += ["", note]

    warned = [tag_of(c) for c in CLIP_GRID
              if (_val(res.get(tag_of(c), {}), "quant_warn_rounds") or 0) > 0]
    if warned:
        lines += ["", f"> **float32 confound:** arms {', '.join(warned)} tripped the "
                      "transmission-quantization warning (client.py): part of the "
                      "clipped update fell below the float32 quantum on the "
                      "large-|g| trap layer. eps still holds (post-processing), but "
                      "those arms measure rounding as well as DP."]

    lines += show_lines

    results_md = OUT / "RESULTS.md"
    results_md.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {results_md}")


if __name__ == "__main__":
    main()

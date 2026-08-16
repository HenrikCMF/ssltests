import torch

import regroup_fragments as RG


class result_analytics():
    def __init__(self):
        pass

    def fragment_decode(self, path, n, seed=None, sign_canon=True,
                        skip_blank=True):
        """Decode `n` randomly chosen images out of one round-major fragment file.

        `path` is a file written by attacks.Loki.save_round_fragments, e.g.
        "fragments_eps_1e+26/round_002.pt". It holds that whole round's raw
        fragments as {"frags": [N,3,32,32] float32, "bins": [N], "counts": [N]}
        plus, for runs extracted after the sign fix, "cutoffs": [N].

        What comes out of the trap is NOT yet a viewable image: a fragment is
        dL/dh_i * x_view / max|.|, i.e. signed and scaled to max-abs 1. Two steps
        of the extraction pipeline turn it into a picture, and both are applied
        here so the result matches what regroup_fragments feeds the encoder:

          1. Sign canonicalisation. dL/dh_i has no fixed sign under BYOL's
             symmetrised loss, so ~half of every round is stored as a photographic
             negative. Done over the FULL round, not the sample, because the sign
             reference is fitted from the round's own bin/cutoff statistics
             (regroup_fragments._sign_reference) and a subset would not pin it.
             Rounds already at the DP noise floor cannot be canonicalised and are
             left as-is -- the helper prints which case each file hit.
          2. Per-image min-max to [0, 1] (regroup_fragments.to_unit).

        Args:
          path:       round file to read.
          n:          how many images to return, sampled WITHOUT replacement.
          seed:       int for a reproducible draw; None (default) draws fresh.
          sign_canon: False returns the raw stored signs, so roughly half the
                      images come back as negatives (the pre-fix behaviour).
          skip_blank: True (default) samples only fragments the real pipeline
                      would keep -- per-fragment std >= regroup_fragments
                      .BLANK_STD, which drops the bins that never fired. False
                      samples the file's whole contents, blanks included.

        Returns a float32 tensor [n, 3, 32, 32] with values in [0, 1], ready for
        torchvision.utils.save_image or matplotlib's imshow (after a permute to
        HWC). Raises ValueError if the file cannot supply `n` distinct images.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")

        obj = torch.load(path, map_location="cpu", weights_only=False)
        frags = (obj["frags"] if isinstance(obj, dict) else obj).float()
        if frags.numel() == 0:
            raise ValueError(f"{path} holds no fragments")

        if sign_canon and isinstance(obj, dict):
            frags = RG._canonicalise(path, obj, frags)

        if skip_blank:
            usable = (frags.flatten(1).std(1) >= RG.BLANK_STD).nonzero(as_tuple=True)[0]
        else:
            usable = torch.arange(frags.shape[0])
        if usable.numel() < n:
            raise ValueError(
                f"{path} has only {usable.numel()} usable fragments "
                f"({frags.shape[0]} stored"
                f"{', blanks skipped' if skip_blank else ''}), cannot draw {n}"
            )

        gen = None
        if seed is not None:
            gen = torch.Generator().manual_seed(int(seed))
        pick = usable[torch.randperm(usable.numel(), generator=gen)[:n]]
        return RG.to_unit(frags[pick])

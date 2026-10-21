"""One-shot diagnostics: how spread-out are FedBYOL latents vs a collapsed / random-init baseline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T

from architectures import ResNet18Projv3
from dataloader import CIFAR10BYOLClientData as C
from result_evaluation import load_encoder
from server import knn_acc

CKPT = Path(__file__).resolve().parents[1] / "eval_model_eps_1e+26.pth"
DATA = Path(__file__).resolve().parents[1] / "data"
N_PAIR = 4096  # two-view pairs
N_COS = 10000  # pairwise cosine sample (full CIFAR-10 test)


def participation_ratio(z):
    """Effective dimensionality of centered features. Collapse -> 1."""
    zc = z - z.mean(0, keepdim=True)
    # C is [D, D]; eigvals of Gram via covariance
    cov = (zc.T @ zc) / max(1, zc.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov.float()).clamp_min(0)
    s1 = eig.sum()
    s2 = (eig * eig).sum()
    return float((s1 * s1) / (s2 + 1e-12)), eig


@torch.no_grad()
def extract(model, loader, device, which="backbone"):
    hs, zs, ys = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        h = model.encode_backbone(x, normalize=False).float()
        z = model(x, normalize=False).float()
        hs.append(h.cpu())
        zs.append(z.cpu())
        ys.append(y.clone())
    return torch.cat(hs), torch.cat(zs), torch.cat(ys)


def cosine_breakdown(z, y, max_n=None):
    """Mean off-diagonal cosine, same-class vs different-class."""
    if max_n is not None and z.shape[0] > max_n:
        z, y = z[:max_n], y[:max_n]
    zn = F.normalize(z.float(), dim=1)
    G = zn @ zn.T
    n = G.shape[0]
    eye = torch.eye(n, dtype=torch.bool)
    same = y[:, None].eq(y[None, :])
    off = ~eye
    def mean_where(mask):
        return float(G[mask].mean()) if mask.any() else float("nan")
    # uniformity: log E exp(-2 ||zi-zj||^2) on the sphere; ||zi-zj||^2 = 2-2cos
    t = 2.0
    sq = (2 - 2 * G[off]).clamp_min(0)
    unif = float(torch.log(torch.exp(-t * sq).mean().clamp_min(1e-12)))
    # quantile() rejects a 100M-entry view; sample off-diagonals instead
    sample = G[off]
    if sample.numel() > 2_000_000:
        sample = sample[torch.randperm(sample.numel())[:2_000_000]]
    return {
        "mean_cos_all": mean_where(off),
        "mean_cos_same_class": mean_where(off & same),
        "mean_cos_diff_class": mean_where(off & ~same),
        "median_cos_all": float(sample.median()),
        "p05_cos": float(sample.quantile(0.05)),
        "p95_cos": float(sample.quantile(0.95)),
        "uniformity": unif,
        "mean_cos_to_mean": float((zn * F.normalize(zn.mean(0, keepdim=True), dim=1)).sum(-1).mean()),
    }


def raw_stats(z):
    norms = z.float().norm(dim=1)
    std_dim = z.float().std(dim=0)
    return {
        "mean_l2": float(norms.mean()),
        "std_l2": float(norms.std()),
        "mean_per_dim_std": float(std_dim.mean()),
        "median_per_dim_std": float(std_dim.median()),
        "frac_dead_dims_std<1e-3": float((std_dim < 1e-3).float().mean()),
        "frac_dead_dims_std<1e-2": float((std_dim < 1e-2).float().mean()),
    }


def report(name, h, z, y):
    print(f"\n===== {name}  N={h.shape[0]}  backbone={tuple(h.shape[1:])}  proj={tuple(z.shape[1:])} =====")
    for tag, feat in (("backbone", h), ("projector", z)):
        rs = raw_stats(feat)
        pr, eig = participation_ratio(feat)
        pr_n, eig_n = participation_ratio(F.normalize(feat, dim=1))
        cb = cosine_breakdown(feat, y, max_n=N_COS)
        # how many eigenvalues carry 90% of variance
        e = eig.flip(0)  # eigvalsh is ascending
        c = torch.cumsum(e, 0) / e.sum().clamp_min(1e-12)
        k90 = int((c < 0.90).sum().item()) + 1
        print(f"  [{tag:10s}] raw mean||z||={rs['mean_l2']:.3f}±{rs['std_l2']:.3f}  "
              f"mean per-dim std={rs['mean_per_dim_std']:.4f}  "
              f"dead(<1e-2)={100*rs['frac_dead_dims_std<1e-2']:.1f}%")
        print(f"  [{tag:10s}] effective dim (raw)={pr:.1f}/{feat.shape[1]}  "
              f"(L2-normed)={pr_n:.1f}  dims for 90% var={k90}")
        print(f"  [{tag:10s}] mean cos all={cb['mean_cos_all']:+.3f}  "
              f"same-class={cb['mean_cos_same_class']:+.3f}  "
              f"diff-class={cb['mean_cos_diff_class']:+.3f}  "
              f"p05={cb['p05_cos']:+.3f} p95={cb['p95_cos']:+.3f}")
        print(f"  [{tag:10s}] cos-to-mean-dir={cb['mean_cos_to_mean']:+.3f}  "
              f"uniformity={cb['uniformity']:+.3f}  (collapse would be 0)")


@torch.no_grad()
def two_view_alignment(model, device, n=N_PAIR):
    aug = T.Compose([
        T.RandomResizedCrop(C.IMG_SIZE, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize(mean=C.MEAN, std=C.STD),
    ])
    raw = datasets.CIFAR10(str(DATA), train=True, download=False, transform=None)
    xs1, xs2, ys = [], [], []
    for i in range(n):
        img, y = raw[i]
        xs1.append(aug(img))
        xs2.append(aug(img))
        ys.append(y)
    x1 = torch.stack(xs1).to(device)
    x2 = torch.stack(xs2).to(device)
    y = torch.tensor(ys)

    def align(encode):
        a = F.normalize(encode(x1).float(), dim=1)
        b = F.normalize(encode(x2).float(), dim=1)
        pos = (a * b).sum(-1)
        # shuffle negatives
        perm = torch.randperm(n, device=device)
        neg = (a * b[perm]).sum(-1)
        return float(pos.mean()), float(pos.median()), float(neg.mean()), float((pos > neg).float().mean())

    pos_h, med_h, neg_h, acc_h = align(lambda x: model.encode_backbone(x, normalize=False))
    pos_z, med_z, neg_z, acc_z = align(lambda x: model(x, normalize=False))
    print("\n===== two random BYOL views of the same CIFAR-10 train image =====")
    print(f"  backbone  pos cos mean={pos_h:+.3f} median={med_h:+.3f}  "
          f"shuffled-pair={neg_h:+.3f}  pos>neg={100*acc_h:.1f}%")
    print(f"  projector pos cos mean={pos_z:+.3f} median={med_z:+.3f}  "
          f"shuffled-pair={neg_z:+.3f}  pos>neg={100*acc_z:.1f}%")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  ckpt={CKPT}")
    trained = load_encoder(str(CKPT), device)
    rand = ResNet18Projv3(emb_dim=2048).to(device).eval()

    _, test_ld = C.build_eval_loaders(data_dir=str(DATA), batch_size=512, num_workers=2)
    h_t, z_t, y = extract(trained, test_ld, device)
    h_r, z_r, y_r = extract(rand, test_ld, device)
    assert torch.equal(y, y_r)

    report("TRAINED eval_model_eps_1e+26.pth  (CIFAR-10 test, center-crop)", h_t, z_t, y)
    report("RANDOM INIT ResNet18Projv3  (same images)", h_r, z_r, y)

    # kNN on L2-normalized backbone, same protocol as server.py eval
    train_ld, _ = C.build_eval_loaders(data_dir=str(DATA), batch_size=512, num_workers=2)
    h_tr, _, y_tr = extract(trained, train_ld, device)
    knn = knn_acc(
        F.normalize(h_tr, dim=1).to(device), y_tr.to(device),
        F.normalize(h_t, dim=1).to(device), y.to(device),
    )
    print(f"\n===== kNN (backbone, k=200, cosine) trained={100*knn:.2f}%  "
          f"chance=10%  r0-random-init typically ~36-38% =====")

    two_view_alignment(trained, device)
    print("\n(random-init two-view, same protocol)")
    two_view_alignment(rand, device)


if __name__ == "__main__":
    main()

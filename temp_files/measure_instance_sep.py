"""Instance-level separation of the FedBYOL backbone, the cut regroup_fragments uses."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms as T

from dataloader import CIFAR10BYOLClientData as C
from result_evaluation import load_encoder

CKPT = Path(__file__).resolve().parents[1] / "eval_model_eps_1e+26.pth"
DATA = Path(__file__).resolve().parents[1] / "data"
N_IMG = 1024
N_VIEWS = 8
EPS = 0.9


def aug():
    return T.Compose([
        T.RandomResizedCrop(C.IMG_SIZE, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize(mean=C.MEAN, std=C.STD),
    ])


@torch.no_grad()
def main():
    device = torch.device("cuda")
    model = load_encoder(str(CKPT), device)
    raw = datasets.CIFAR10(str(DATA), train=True, download=False, transform=None)
    tfm = aug()

    xs, ids, ys = [], [], []
    for i in range(N_IMG):
        img, y = raw[i]
        for _ in range(N_VIEWS):
            xs.append(tfm(img))
            ids.append(i)
            ys.append(y)
    x = torch.stack(xs).to(device)
    ids = torch.tensor(ids, device=device)
    ys = torch.tensor(ys, device=device)

    z = []
    for s in range(0, x.shape[0], 256):
        z.append(F.normalize(model.encode_backbone(x[s:s + 256], normalize=False).float(), dim=1))
    z = torch.cat(z, 0)
    G = z @ z.T
    n = G.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=device)
    same_id = ids[:, None].eq(ids[None, :]) & ~eye
    diff_id = ~ids[:, None].eq(ids[None, :])
    same_cls_diff_id = ys[:, None].eq(ys[None, :]) & diff_id
    diff_cls = ~ys[:, None].eq(ys[None, :])

    def stats(mask, name):
        v = G[mask]
        n = int(mask.sum())
        mean = float(v.mean())
        ge80 = float((v >= 0.80).float().mean())
        ge90 = float((v >= 0.90).float().mean())
        ge93 = float((v >= 0.93).float().mean())
        if v.numel() > 2_000_000:
            v = v[torch.randint(0, v.numel(), (2_000_000,), device=v.device)]
        print(f"  {name:28s}  n={n:8d}  mean={mean:+.3f}  "
              f"p50={v.median():+.3f}  p05={v.quantile(0.05):+.3f}  p95={v.quantile(0.95):+.3f}  "
              f"frac>=0.80={100*ge80:5.2f}%  "
              f">=0.90={100*ge90:5.2f}%  "
              f">=0.93={100*ge93:5.2f}%")

    print(f"N_img={N_IMG} views/img={N_VIEWS}  backbone L2-normed  (BYOL train augs)")
    stats(same_id, "same original (other views)")
    stats(diff_id, "different original")
    stats(same_cls_diff_id, "same class, different orig")
    stats(diff_cls, "different class")

    # per-view: nearest foreign vs nearest other view of self
    G_self = G.masked_fill(~same_id, -2)
    G_for = G.masked_fill(~diff_id, -2)
    best_self = G_self.max(1).values
    best_for = G_for.max(1).values
    print("\nper-view nearest neighbour:")
    print(f"  nearest other view of SAME original: mean={best_self.mean():+.3f}  "
          f"median={best_self.median():+.3f}  frac>=0.90={(best_self>=0.90).float().mean()*100:.1f}%")
    print(f"  nearest DIFFERENT original:          mean={best_for.mean():+.3f}  "
          f"median={best_for.median():+.3f}  frac>=0.90={(best_for>=0.90).float().mean()*100:.1f}%")
    print(f"  same-orig NN beats foreign NN:       {(best_self > best_for).float().mean()*100:.1f}%")

    # kNN purity at k=99 and at the DBSCAN edge (cos>=0.9, up to 99)
    k = 99
    G_off = G.masked_fill(eye, -2)
    topv, topi = G_off.topk(k, dim=1)
    neigh_same = ids[topi].eq(ids[:, None])
    print(f"\nk={k} NN (no self): mean fraction same-original = {neigh_same.float().mean()*100:.1f}%")
    print(f"  mean cosine of those k NN: {topv.mean():+.3f}  min-of-k mean={topv[:, -1].mean():+.3f}")

    edge = topv >= EPS
    n_edge = edge.float().sum(1)
    pure = torch.where(n_edge > 0, (neigh_same & edge).float().sum(1) / n_edge.clamp_min(1), torch.zeros_like(n_edge))
    print(f"\nDBSCAN neighbourhood at eps={EPS} (among the k={k} NN):")
    print(f"  mean #neighbours >= {EPS}: {n_edge.mean():.2f}  "
          f"frac of views with >=3 such: {(n_edge>=3).float().mean()*100:.1f}%")
    print(f"  mean purity of those neighbours: {pure[n_edge>0].mean()*100:.1f}%  "
          f"(same original)")
    print(f"  frac of >=0.9 edges that are same-original: "
          f"{((neigh_same & edge).sum() / edge.sum().clamp_min(1))*100:.1f}%")


if __name__ == "__main__":
    main()

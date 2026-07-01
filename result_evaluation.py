"""
Linear-evaluation protocol on a trained FedSSL encoder (eval_model.pth).

This reproduces the paper's linear-probe protocol (Zhuang et al., ICLR 2022,
Appendix B.3) so we can compare apples-to-apples against their headline FedBYOL
number of 79.44% top-1 (which is *linear eval*, NOT the kNN monitoring curve).

Paper protocol:
  - Freeze the trained encoder. The "representation" is the backbone output with
    the two-layer projection MLP removed (== our ResNet18Projv3.encode_backbone,
    512-d).
  - Train a single new fully-connected layer on top of the frozen features for
    200 epochs, batch size 512, Adam, lr 3e-3.
  - Report top-1 accuracy on the CIFAR-10 test set.

Run:  python result_evaluation.py
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from architectures import ResNet18Projv3
from dataloader import CIFAR10BYOLClientData

EMBEDDING_SIZE = 2048          # must match training (FedEMA_run.py)
CKPT = "eval_model.pth"


def load_encoder(ckpt_path, device):
    model = ResNet18Projv3(emb_dim=EMBEDDING_SIZE)
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and not any(k.startswith("backbone") for k in sd):
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[warn] missing keys when loading {ckpt_path}: {missing}")
    if unexpected:
        print(f"[warn] unexpected keys when loading {ckpt_path}: {unexpected}")
    return model.to(device).eval()


@torch.no_grad()
def extract_feats(model, loader, device, normalize=False):
    """Frozen 512-d backbone features (projection MLP removed = paper identity)."""
    feats, labels = [], []
    use_amp = str(device).startswith("cuda")
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            z = model.encode_backbone(x, normalize=normalize)   # [B, 512]
        feats.append(z.float().cpu())
        labels.append(y.clone())
    return torch.cat(feats, 0), torch.cat(labels, 0)


def linear_probe(train_feats, train_labels, test_feats, test_labels, device,
                 num_epochs=200, lr=3e-3, batch_size=512, weight_decay=0.0,
                 standardize=False, log_every=20):
    """Paper protocol: 1 FC layer, Adam, lr 3e-3, 200 epochs, batch 512."""
    train_feats = train_feats.clone()
    test_feats = test_feats.clone()

    if standardize:
        mu = train_feats.mean(0, keepdim=True)
        sigma = train_feats.std(0, keepdim=True) + 1e-6
        train_feats = (train_feats - mu) / sigma
        test_feats = (test_feats - mu) / sigma

    feat_dim = train_feats.shape[1]
    num_classes = int(train_labels.max().item()) + 1

    clf = nn.Linear(feat_dim, num_classes).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()

    train_ds = TensorDataset(train_feats, train_labels)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    test_feats_d = test_feats.to(device)
    test_labels_d = test_labels.to(device)

    best = 0.0
    for ep in range(1, num_epochs + 1):
        clf.train()
        for f, y in train_ld:
            f, y = f.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(clf(f), y)
            loss.backward()
            opt.step()

        clf.eval()
        with torch.no_grad():
            acc = (clf(test_feats_d).argmax(1) == test_labels_d).float().mean().item()
        best = max(best, acc)
        if ep % log_every == 0 or ep == 1 or ep == num_epochs:
            print(f"  epoch {ep:3d}/{num_epochs} | test top-1 {acc*100:5.2f}% | best {best*100:5.2f}%")

    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--data-dir", default="./data")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | checkpoint: {args.ckpt}")

    model = load_encoder(args.ckpt, device)
    train_ld, test_ld = CIFAR10BYOLClientData.build_eval_loaders(data_dir=args.data_dir, batch_size=512, num_workers=4)

    # Extract frozen features once (encoder is fixed during linear eval).
    train_feats, train_labels = extract_feats(model, train_ld, device, normalize=False)
    test_feats, test_labels = extract_feats(model, test_ld, device, normalize=False)
    print(f"features: train {tuple(train_feats.shape)} | test {tuple(test_feats.shape)}")

    print("\n=== Linear eval (paper protocol: raw 512-d feats, Adam lr 3e-3, "
          f"{args.epochs} ep, bs {args.batch_size}) ===")
    acc_raw = linear_probe(train_feats, train_labels, test_feats, test_labels, device,
                           num_epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                           standardize=False)

    print("\n=== Cross-check: same protocol on standardized feats ===")
    acc_std = linear_probe(train_feats, train_labels, test_feats, test_labels, device,
                           num_epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                           standardize=True)

    print("\n" + "=" * 56)
    print(f"  Linear eval top-1 (raw feats)          : {acc_raw*100:5.2f}%")
    print(f"  Linear eval top-1 (standardized feats) : {acc_std*100:5.2f}%")
    print(f"  Paper FedBYOL linear eval (CIFAR-10)    : 79.44%")
    print("=" * 56)


if __name__ == "__main__":
    main()

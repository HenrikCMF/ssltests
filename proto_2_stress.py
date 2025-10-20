# file: hirise_ssl_prototypes.py
# ============================================================
# Self-supervised pseudo-labeling on HiRISE
# - SSL (SimCLR-lite with a small negatives queue)
# - Multi-prototype (spherical k-means) nearest-centroid labeling
# - Margin-based abstention, macro-F1, coverage
# ============================================================

import os, time, random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")



PIN_MEMORY = (DEVICE.type == "mps")
# -----------------------
# Config
# -----------------------
SEED = 1337
ROOT = "hirise-map-proj-v3" 
TEST_FRAC = 0.70

# SSL/training
EPOCHS_SSL = 50
BATCH_SSL  = 1024
LR_SSL     = 3e-4 * (BATCH_SSL/128)
WEIGHT_DECAY = 1e-5#1e-4
EMBED_DIM  = 128
TAU        = 0.15              # temperature for contrastive loss
QUEUE_K    = 8192             # negatives memory
IMG_SIZE   = 128

# Prototypes / pseudo-labeling
PER_CLASS_LABELED = 40        # small labeled seed per class
PROTOS_PER_CLASS  = 20        # spherical k-means centers per class
MARGIN_TH         = 0      # accept only if (top1 - top2) >= MARGIN_TH

# Embedding / dataloaders
BATCH_EMB = 1024
NUM_WORKERS = 4

# Device


def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------
# SSL augmentations (operate on PIL; produce 3ch tensors)
# -----------------------
to_3ch = transforms.Lambda(lambda x: x.expand(3, -1, -1))
augment_ssl = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomRotation((0, 360)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.Grayscale(num_output_channels=3),   # <-- picklable, no lambdas
    transforms.ToTensor(),
    transforms.Normalize([0.45]*3, [0.225]*3),
])

class TwoViews(torch.utils.data.Dataset):
    """Wrap a PIL dataset and return two independently augmented views."""
    def __init__(self, base_ds, t1, t2):
        self.ds = base_ds
        self.t1 = t1
        self.t2 = t2
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        img, _ = self.ds[i]      # label ignored in SSL
        return self.t1(img), self.t2(img)

# -----------------------
# Encoder: ResNet-18 + 2-layer projector (ImageNet weights)
# -----------------------
class ResNet18Proj(nn.Module):
    def __init__(self, emb_dim=EMBED_DIM):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        base.fc = nn.Identity()
        self.backbone = base
        self.proj = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(inplace=True),
            nn.Linear(512, emb_dim)
        )
    def forward(self, x, normalize=True):
        h = self.backbone(x)
        z = self.proj(h)
        return F.normalize(z, dim=1) if normalize else z


def nt_xent_inbatch(z1, z2, tau=0.5):
    # z1,z2: [B,D], L2-normalized
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)            # [2B,D]
    sim = z @ z.t()                            # cosine
    mask = torch.eye(2*B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e4)
    # positives are (i, i+B) and (i+B, i)
    pos = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)]).to(z.device)
    logits = sim / tau
    labels = pos
    return F.cross_entropy(logits, labels)

# -----------------------
# Embedding and prototypes
# -----------------------
@torch.no_grad()
def embed_dataset(model, dataset, batch=BATCH_EMB):
    loader = DataLoader(dataset, batch_size=batch, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type=="cuda"))
    Z, Y = [], []
    model.eval()
    for x, y in loader:
        x = x.to(DEVICE)
        z = model(x)                   # normalized
        Z.append(z.cpu())
        Y.append(y)
    return torch.cat(Z, 0), torch.cat(Y, 0)

def spherical_kmeans(X, k, iters=20, seed=SEED):
    """
    X: [N,D] L2-normalized.
    Returns centers [k,D] and assignments [N].
    """
    rng = torch.Generator().manual_seed(seed)
    N, D = X.size()
    idx = torch.randperm(N, generator=rng)[:k]
    C = X[idx].clone()  # [k,D]
    for _ in range(iters):
        sims = X @ C.t()                 # [N,k]
        assign = sims.argmax(1)
        newC = []
        for j in range(k):
            sel = X[assign == j]
            if sel.numel() == 0:
                newC.append(C[j:j+1])
            else:
                newC.append(F.normalize(sel.mean(0, keepdim=True), dim=1))
        C = torch.cat(newC, 0)
    return C, assign

def make_multi_prototypes(emb, lab, k_per_class=PROTOS_PER_CLASS):
    protos, proto_labels = [], []
    classes = torch.unique(lab).tolist()
    for cls in classes:
        X = F.normalize(emb[lab == cls], dim=1)
        k = min(k_per_class, X.size(0))
        C, _ = spherical_kmeans(X, k)
        protos.append(C)
        proto_labels += [int(cls)] * k
    P = torch.cat(protos, dim=0)             # [sum_k, D]
    yP = torch.tensor(proto_labels)
    return P, yP

@torch.no_grad()
def nearest_proto_predict(emb, P, yP, margin_th=MARGIN_TH):
    sims = emb @ P.t()                       # cosine
    top2 = sims.topk(k=2, dim=1)
    pred_idx = top2.indices[:, 0]
    margins = top2.values[:, 0] - top2.values[:, 1]
    y_pred = yP[pred_idx]
    accept_mask = margins >= margin_th
    # Set abstained labels to -1
    y_out = y_pred.clone()
    y_out[~accept_mask] = -1
    return y_out, margins, accept_mask

def macro_f1(y_true, y_pred, ignore_label=-1):
    """
    Macro-F1 over classes present in y_true, ignoring predictions == ignore_label.
    """
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    mask = (y_pred != ignore_label)
    if mask.sum() == 0:
        return 0.0
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    classes = sorted(set(y_true.tolist()))
    f1s = []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        f1s.append(f1)
    return float(np.mean(f1s))

# -----------------------
# Main
# -----------------------
def main():
    
    print(f"Device: {DEVICE}")
    set_seed()
    # 1) Build datasets from disk (hd handles parsing + eval transforms)
    train_tensor, test_tensor, train_pil, id2name = hd.build_hirise_splits(
        ROOT, test_frac=TEST_FRAC, img_size=IMG_SIZE
    )
    ntrain, ntest = len(train_tensor), len(test_tensor)
    nclasses = len(id2name)
    print(f"Loaded HiRISE: train={ntrain}, test={ntest}, classes={nclasses}")

    # 2) Build a small labeled seed (balanced per class) and unlabeled pool (rest of train)
    labels_train = [y for _, y in train_tensor.samples]  # from dataset internals
    server_idx = []
    for c in sorted(set(labels_train)):
        idx_c = [i for i, (_, y) in enumerate(train_tensor.samples) if y == c]
        server_idx.extend(idx_c[:min(PER_CLASS_LABELED, len(idx_c))])
    server_idx = sorted(server_idx)

    mask = torch.zeros(len(train_tensor), dtype=torch.bool)
    mask[server_idx] = True
    client_idx = [i for i in range(len(train_tensor)) if not mask[i]]

    server_labeled   = Subset(train_tensor, server_idx)
    server_ssl_base  = Subset(train_pil,   server_idx)
    client_unlabeled = Subset(train_tensor, client_idx)

    print(f"Labeled seed: {len(server_labeled)} samples "
          f"({PER_CLASS_LABELED} per class if available). "
          f"Client unlabeled pool: {len(client_unlabeled)}")

    # 3) SSL pretrain on the small seed (two views)
    ssl_ds = TwoViews(train_pil, augment_ssl, augment_ssl)#ssl_ds = TwoViews(server_ssl_base, augment_ssl, augment_ssl)
    NUM_WORKERS = 4
    ssl_loader = DataLoader(
        ssl_ds, batch_size=BATCH_SSL, shuffle=True, drop_last=True,
        num_workers=NUM_WORKERS, persistent_workers=True, prefetch_factor=2
    )

    encoder = ResNet18Proj(emb_dim=EMBED_DIM).to(DEVICE)
    opt = torch.optim.AdamW(encoder.parameters(), lr=LR_SSL, weight_decay=WEIGHT_DECAY)
    #fq = FeatureQueue(dim=EMBED_DIM, K=QUEUE_K, device=DEVICE)

    encoder.train()
    t0 = time.time()
    for ep in range(1, EPOCHS_SSL + 1):
        total = 0.0
        for x1, x2 in ssl_loader:
            x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
            with torch.autocast(device_type="mps", dtype=torch.float16):
                z1, z2 = encoder(x1), encoder(x2)
                loss = nt_xent_inbatch(z1, z2, tau=TAU)
            #z1, z2 = encoder(x1), encoder(x2)
            #loss = nt_xent_inbatch(z1, z2, tau=TAU)
            opt.zero_grad()
            loss.backward()
            opt.step()
            #with torch.no_grad():
            #    fq.enqueue(z2)
            total += loss.item()
        if ep % 10 == 0 or ep == 1:
            print(f"[SSL] epoch {ep:03d}/{EPOCHS_SSL}  loss={total/len(ssl_loader):.4f}")
    print(f"SSL time: {time.time()-t0:.1f}s")
    encoder.eval()

    # 4) Build multi-prototypes from the labeled seed
    emb_server, y_server = embed_dataset(encoder, server_labeled, batch=BATCH_EMB)
    P, yP = make_multi_prototypes(emb_server, y_server, k_per_class=PROTOS_PER_CLASS)
    print(f"Built prototypes: {P.size(0)} total across {nclasses} classes.")

    # 5) Pseudo-label client pool (evaluate with GT for analysis)
    emb_client, y_client_true = embed_dataset(encoder, client_unlabeled, batch=BATCH_EMB)
    y_client_pred, margins_client, accept_client = nearest_proto_predict(
        emb_client, P, yP, margin_th=MARGIN_TH
    )
    cov = float(accept_client.float().mean().item())  # coverage
    acc = float((y_client_pred[accept_client] == y_client_true[accept_client]).float().mean().item()) if cov > 0 else 0.0
    f1  = macro_f1(y_client_true, y_client_pred, ignore_label=-1)
    print(f"[Client pseudo-labeling] coverage={cov*100:.1f}%  acc_on_accepted={acc*100:.2f}%  macroF1(accepted)={f1*100:.2f}%")
    print(f"Margin: mean={margins_client.mean().item():.3f}, median={margins_client.median().item():.3f}")

    # 6) Also report nearest-prototype performance on the held-out TEST set
    emb_test, y_test = embed_dataset(encoder, test_tensor, batch=BATCH_EMB)
    y_test_pred, margins_test, accept_test = nearest_proto_predict(
        emb_test, P, yP, margin_th=MARGIN_TH
    )
    cov_t = float(accept_test.float().mean().item())
    acc_t = float((y_test_pred[accept_test] == y_test[accept_test]).float().mean().item()) if cov_t > 0 else 0.0
    f1_t  = macro_f1(y_test, y_test_pred, ignore_label=-1)
    print(f"[Test set] coverage={cov_t*100:.1f}%  acc_on_accepted={acc_t*100:.2f}%  macroF1(accepted)={f1_t*100:.2f}%")
    print(f"Done. Classes: {id2name}")

if __name__ == "__main__":
    main()

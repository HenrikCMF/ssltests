import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split, SubsetRandomSampler


class ReadoutLabeler:
    def __init__(self, device, batch_emb=1024, num_workers=4, pin_memory=False, seed=42):
        self.DEVICE = device
        self.BATCH_EMB = batch_emb
        self.NUM_WORKERS = num_workers
        self.PIN_MEMORY = pin_memory
        self.SEED = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    # ---------- Embeddings ----------
    @torch.no_grad()
    def embed_dataset(self, model, dataset, batch=1024, use_backbone=False):
        loader = DataLoader(dataset, batch_size=batch, shuffle=False,
                            num_workers=self.NUM_WORKERS, pin_memory=self.PIN_MEMORY)
        Z, Y = [], []
        model.eval()
        for x, y in loader:
            x = x.to(self.DEVICE)
            if use_backbone and hasattr(model, "encode_backbone"):
                z = model.encode_backbone(x)       # 512-d features, L2-normalized
            else:
                z = model(x)                       # projected z (your current default)
            Z.append(z.cpu())
            Y.append(y)
        return torch.cat(Z, 0), torch.cat(Y, 0)

    # ---------- Prototypes (your current path) ----------
    def spherical_kmeans(self, X, k, iters=20):
        rng = torch.Generator().manual_seed(self.SEED)
        N, D = X.size()
        idx = torch.randperm(N, generator=rng)[:k]
        C = X[idx].clone()  # [k,D]
        for _ in range(iters):
            sims = X @ C.t()           # [N,k]
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

    def make_multi_prototypes(self, emb, lab, k_per_class=20):
        protos, labels = [], []
        classes = torch.unique(lab).tolist()
        for c in classes:
            X = F.normalize(emb[lab == c], dim=1)
            k = min(k_per_class, X.size(0))
            C, _ = self.spherical_kmeans(X, k)
            protos.append(C)
            labels += [int(c)] * k
        P  = torch.cat(protos, 0) if len(protos)>0 else torch.empty(0, emb.size(1))
        yP = torch.tensor(labels, dtype=torch.long)
        return P, yP

    @torch.no_grad()
    def predict_prototype(self, emb, P, yP, margin_th=0.0):
        sims   = emb @ P.t()                        # [N, K]
        top2   = sims.topk(k=2, dim=1)
        pred_i = top2.indices[:,0]
        margins = top2.values[:,0] - top2.values[:,1]
        y_pred = yP[pred_i]
        accept = margins >= margin_th
        y_out  = y_pred.clone()
        y_out[~accept] = -1
        return y_out, margins, accept

    # ---------- Cosine Linear Probe ----------
    def fit_linear_probe(self, Z_seed, y_seed, nclasses, epochs=30, lr=5e-3):
        # weight-normalized cosine classifier
        head = nn.Linear(Z_seed.size(1), nclasses, bias=False).to(self.DEVICE)
        opt  = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)

        Zs = Z_seed.to(self.DEVICE)
        ys = y_seed.to(self.DEVICE)
        for _ in range(epochs):
            W = F.normalize(head.weight, dim=1)          # [C,D]
            logits = (Zs @ W.t())                        # [N,C]
            loss = F.cross_entropy(logits, ys)
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        return head

    @torch.no_grad()
    def predict_linear(self, Z, head):
        W = F.normalize(head.weight, dim=1)             # [C,D]
        logits = Z.to(self.DEVICE) @ W.t()
        return logits.argmax(1).cpu()

    # ---------- k-NN (soft, temperature-weighted) ----------
    @torch.no_grad()
    def knn_predict(self, Z_train, y_train, Z_query, k=200, T=0.07):
        # Z_* assumed L2-normalized already
        sims = Z_query @ Z_train.t()                    # [Q,Nt]
        topk = sims.topk(k, dim=1)
        W    = torch.exp(topk.values / T)               # [Q,k]
        Yk   = y_train[topk.indices]                    # [Q,k]
        C = int(y_train.max().item()) + 1
        out = torch.zeros(Z_query.size(0), C)
        out.scatter_add_(1, Yk.cpu(), W.cpu())
        return out.argmax(1)

class clustering_labeler:
    def __init__(self, BATCH_EMB=1024, PROTOS_PER_CLASS=20, MARGIN_TH=0.0, NUM_WORKERS=4, SEED=42):
        if torch.cuda.is_available():
            DEVICE = torch.device("cuda")
        elif torch.backends.mps.is_available():
            DEVICE = torch.device("mps")
        else:
            DEVICE = torch.device("cpu")
        self.PIN_MEMORY = (DEVICE.type == "cuda")
        self.DEVICE = DEVICE
        self.BATCH_EMB = BATCH_EMB
        self.PROTOS_PER_CLASS = PROTOS_PER_CLASS
        self.MARGIN_TH = MARGIN_TH
        self.NUM_WORKERS = NUM_WORKERS
        self.SEED = SEED

    @torch.no_grad()
    def embed_dataset(self, model, dataset, batch=1024):
        loader = DataLoader(dataset, batch_size=batch, shuffle=False,
                            num_workers=self.NUM_WORKERS, pin_memory=self.PIN_MEMORY)
        Z, Y = [], []
        model.eval()
        for x, y in loader:
            x = x.to(self.DEVICE)
            z = model(x)                   # normalized
            Z.append(z.cpu())
            Y.append(y)
        return torch.cat(Z, 0), torch.cat(Y, 0)

    def spherical_kmeans(self, X, k, iters=20, seed=42):
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

    def make_multi_prototypes(self, emb, lab, k_per_class=20):
        protos, proto_labels = [], []
        classes = torch.unique(lab).tolist()
        for cls in classes:
            X = F.normalize(emb[lab == cls], dim=1)
            k = min(k_per_class, X.size(0))
            C, _ = self.spherical_kmeans(X, k)
            protos.append(C)
            proto_labels += [int(cls)] * k
        P = torch.cat(protos, dim=0)             # [sum_k, D]
        yP = torch.tensor(proto_labels)
        return P, yP

    @torch.no_grad()
    def nearest_proto_predict(self,emb, P, yP, margin_th=0):
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
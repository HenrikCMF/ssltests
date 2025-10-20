import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from typing import Tuple, List
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split, SubsetRandomSampler
import random
from pathlib import Path
import architectures
import classifier_heads
from torchvision import datasets, transforms, models
import save_model
DATA_DIR = Path(__file__).resolve().parent / "data"   # absolute path

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
    
    


class SSL_module:
    def __init__(self, SEED):
        if torch.cuda.is_available():
            DEVICE = torch.device("cuda")
        elif torch.backends.mps.is_available():
            DEVICE = torch.device("mps")
        else:
            DEVICE = torch.device("cpu")
        self.PIN_MEMORY = (DEVICE.type == "cuda")
        self.DEVICE = DEVICE

        self.SEED = SEED
        self.TEST_FRAC = 0.70

        # SSL/training
        self.EPOCHS_SSL = 400#50
        self.BATCH_SSL  = 512
        self.LR_SSL     = 3e-4 #* (self.BATCH_SSL/128)
        self.WEIGHT_DECAY = 1e-4
        self.EMBED_DIM  = 128
        self.TAU        = 0.2             # temperature for contrastive loss
        self.QUEUE_K    = 8192             # negatives memory
        self.IMG_SIZE   = 32

        # Prototypes / pseudo-labeling
        self.PER_CLASS_LABELED = 40        # small labeled seed per class
        self.PROTOS_PER_CLASS  = 20        # spherical k-means centers per class
        self.MARGIN_TH         = 0      # accept only if (top1 - top2) >= MARGIN_TH

        # Embedding / dataloaders
        self.BATCH_EMB = 1024
        self.NUM_WORKERS = 4

    def set_seed(self, s=42):
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def get_aug_transform(self):
        augment_ssl = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4,0.4,0.4,0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)),
        ])
        return None, augment_ssl
    
    def nt_xent_inbatch(self, z1, z2, tau=0.5):
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
    
    def macro_f1(self, y_true, y_pred, ignore_label=-1):
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
    
    
class Do_SSL:
    def __init__(self, seed, train=True):
        self.obj = SSL_module(seed)
        to_3ch, augment_ssl= self.obj.get_aug_transform()
        self.ssl_loader = self.get_data(DATA_DIR, augment_ssl)
        self.model = architectures.ResNet18Projv2(emb_dim=self.obj.EMBED_DIM).to(self.obj.DEVICE)

        if train:
            opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
            warmup, T = 10, self.obj.EPOCHS_SSL
            def set_lr(ep):
                if ep <= warmup:
                    lr = 1e-3 * ep / warmup
                else:
                    t = (ep-warmup)/max(1,T-warmup)
                    lr = 1e-5 + 0.5*(1e-3-1e-5)*(1+np.cos(np.pi*t))
                for g in opt.param_groups: g['lr'] = lr

            for ep in range(1, T+1):
                set_lr(ep)
                total=0.0
                for x1,x2 in self.ssl_loader:
                    x1,x2 = x1.to(self.obj.DEVICE), x2.to(self.obj.DEVICE)
                    z1,z2 =self.model(x1), self.model(x2)
                    loss = self.obj.nt_xent_inbatch(z1,z2,tau=0.2)
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    opt.step()
                    total += loss.item()
                if ep % 10 == 0 or ep == 1:
                    print(f"[SSL] epoch {ep:03d}/{self.obj.EPOCHS_SSL}  loss={total/len(self.ssl_loader):.4f}")
            self.save_ssl_checkpoint("ssl-final.pt", epoch=ep)
            self.model.eval()
            self.encoder = self.model
            print("Finished SSL training.")
        else:
            #self.model= save_model.load_ssl_checkpoint("ssl-final.pt", self.obj.DEVICE, lambda emb_dim: ResNet18Proj(emb_dim))
            #self.encoder = self.model
            #self.encoder.eval()
            pass


    def get_data(self, dir, augment_ssl):
        img_size = getattr(self.obj, "IMG_SIZE", 32)
        num_workers = getattr(self.obj, "NUM_WORKERS", 4)
        batch_ssl   = getattr(self.obj, "BATCH_SSL", 128)
        pin_memory  = getattr(self.obj, "PIN_MEMORY", False)

        tf = []
        if img_size != 32:
            tf += [transforms.Resize(img_size), transforms.CenterCrop(img_size)]
        tf += [transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))]
        eval_tf = transforms.Compose(tf)

        self.train_tensor = datasets.CIFAR10(root=dir, train=True,  download=True,  transform=eval_tf)
        self.test_tensor  = datasets.CIFAR10(root=dir, train=False, download=True,  transform=eval_tf)
        self.train_pil    = datasets.CIFAR10(root=dir, train=True,  download=False, transform=None)

        class_names = self.train_tensor.classes
        self.id2name = {i:n for i,n in enumerate(class_names)}
        self.ntrain, self.ntest = len(self.train_tensor), len(self.test_tensor)
        self.nclasses = len(self.id2name)

        ssl_ds = TwoViews(self.train_pil, augment_ssl, augment_ssl)
        ssl_loader = DataLoader(
            ssl_ds, batch_size=batch_ssl, shuffle=True, drop_last=True,
            num_workers=num_workers, persistent_workers=(num_workers>0),
            prefetch_factor=(2 if num_workers>0 else None), pin_memory=pin_memory
        )
        return ssl_loader

    
    def evaluate_model(self, ckpt_path="ssl-final.pt",
                   HEAD_TYPE="linear",          # "proto" | "linear" | "knn"
                   PROTOS_PER_CLASS=None,       # falls back to self.obj.PROTOS_PER_CLASS
                   MARGIN_TH=None,              # falls back to self.obj.MARGIN_TH
                   KNN_K=200, KNN_T=0.07):
        # 0) Load encoder from checkpoint (no retraining)
        device = self.obj.DEVICE
        encoder, ep = save_model.load_ssl_encoder(ckpt_path, device, model_ctor=lambda d: architectures.ResNet18Projv2(emb_dim=d))
        if ep is not None:
            print(f"[ckpt] loaded encoder from epoch {ep}")
        self.encoder = encoder

        # 1) Build seed split (same as before, but CIFAR uses .targets)
        labels_train = self.train_tensor.targets
        server_idx = []
        for c in range(self.nclasses):
            idx_c = [i for i,y in enumerate(labels_train) if y == c]
            server_idx.extend(idx_c[:min(self.obj.PER_CLASS_LABELED, len(idx_c))])
        server_idx = sorted(server_idx)
        mask = torch.zeros(len(self.train_tensor), dtype=torch.bool)
        mask[server_idx] = True
        client_idx = [i for i in range(len(self.train_tensor)) if not mask[i]]

        server_labeled   = Subset(self.train_tensor, server_idx)
        client_unlabeled = Subset(self.train_tensor, client_idx)

        # 2) Embed seed, unlabeled, and test
        labeler = classifier_heads.ReadoutLabeler(device=device, batch_emb=self.obj.BATCH_EMB,
                                num_workers=self.obj.NUM_WORKERS, pin_memory=self.obj.PIN_MEMORY, seed=self.obj.SEED)
        Z_seed, y_seed = labeler.embed_dataset(self.encoder, server_labeled,
                                                     batch=self.obj.BATCH_EMB, use_backbone=True)
        Z_cli,  y_cli  = labeler.embed_dataset(self.encoder, client_unlabeled,
                                                            batch=self.obj.BATCH_EMB, use_backbone=True)
        Z_test, y_test = labeler.embed_dataset(self.encoder, self.test_tensor,
                                                            batch=self.obj.BATCH_EMB, use_backbone=True)

        # 3) Choose head and predict
        HEAD_TYPE = HEAD_TYPE.lower()
        if HEAD_TYPE == "proto":
            kpc = PROTOS_PER_CLASS if PROTOS_PER_CLASS is not None else self.obj.PROTOS_PER_CLASS
            P, yP = labeler.make_multi_prototypes(Z_seed, y_seed, k_per_class=kpc)
            mth = self.obj.MARGIN_TH if MARGIN_TH is None else MARGIN_TH

            y_cli_pred, margins_cli, accept_cli = labeler.predict_prototype(Z_cli,  P, yP, margin_th=mth)
            y_tst_pred, margins_t,  accept_t   = labeler.predict_prototype(Z_test, P, yP, margin_th=mth)

        elif HEAD_TYPE == "linear":
            head = labeler.fit_linear_probe(Z_seed, y_seed, nclasses=self.nclasses, epochs=30, lr=5e-3)
            y_cli_pred = labeler.predict_linear(Z_cli, head)
            y_tst_pred = labeler.predict_linear(Z_test, head)
            # For reporting parity with your proto flow:
            margins_cli = torch.zeros(len(y_cli_pred))
            margins_t   = torch.zeros(len(y_tst_pred))
            accept_cli  = torch.ones(len(y_cli_pred), dtype=torch.bool)
            accept_t    = torch.ones(len(y_tst_pred), dtype=torch.bool)

        elif HEAD_TYPE == "knn":
            y_cli_pred = labeler.knn_predict(Z_seed, y_seed, Z_cli,  k=KNN_K, T=KNN_T)
            y_tst_pred = labeler.knn_predict(Z_seed, y_seed, Z_test, k=KNN_K, T=KNN_T)
            margins_cli = torch.zeros(len(y_cli_pred))
            margins_t   = torch.zeros(len(y_tst_pred))
            accept_cli  = torch.ones(len(y_cli_pred), dtype=torch.bool)
            accept_t    = torch.ones(len(y_tst_pred), dtype=torch.bool)

        else:
            raise ValueError(f"Unknown HEAD_TYPE: {HEAD_TYPE}")

        # 4) Metrics (reuse your macro_f1)
        def macro_f1(y_true, y_pred):
            yt = y_true.cpu().numpy(); yp = y_pred.cpu().numpy()
            classes = sorted(set(yt.tolist()))
            f1s = []
            for c in classes:
                tp = np.sum((yt==c)&(yp==c))
                fp = np.sum((yt!=c)&(yp==c))
                fn = np.sum((yt==c)&(yp!=c))
                prec = tp/(tp+fp+1e-12); rec=tp/(tp+fn+1e-12)
                f1s.append(2*prec*rec/(prec+rec+1e-12))
            return float(np.mean(f1s))

        cov_cli = float(accept_cli.float().mean().item())
        acc_cli = float((y_cli_pred[accept_cli]==y_cli[accept_cli]).float().mean().item()) if cov_cli>0 else 0.0
        f1_cli  = macro_f1(y_cli, y_cli_pred)
        print(f"[Client pseudo-labeling][{HEAD_TYPE}] coverage={cov_cli*100:.1f}%  acc={acc_cli*100:.2f}%  macroF1={f1_cli*100:.2f}%")
        if HEAD_TYPE == "proto":
            print(f"Margin: mean={margins_cli.mean().item():.3f}, median={margins_cli.median().item():.3f}")

        cov_t = float(accept_t.float().mean().item())
        acc_t = float((y_tst_pred[accept_t]==y_test[accept_t]).float().mean().item()) if cov_t>0 else 0.0
        f1_t  = macro_f1(y_test, y_tst_pred)
        print(f"[Test set][{HEAD_TYPE}] coverage={cov_t*100:.1f}%  acc={acc_t*100:.2f}%  macroF1={f1_t*100:.2f}%")


    def save_ssl_checkpoint(self, path, epoch):
        """
        Save encoder weights + minimal config so you can reload later.
        """
        ckpt = {
            "epoch": epoch,
            "model_class": self.model.__class__.__name__,
            "state_dict": self.model.state_dict(),          # stays on MPS/CPU fine
            "embed_dim": getattr(self.obj, "EMBED_DIM", None),
        }
        path = Path(path)
        save_model._ensure_dir(path.parent)
        torch.save(ckpt, path)
        print(f"[ckpt] Saved SSL encoder -> {path}")



if __name__ == "__main__":
    ssl_handler=Do_SSL(seed=42, train=False)
    # 1) linear probe (usually a big jump)
    ssl_handler.evaluate_model(ckpt_path="ssl-final.pt", HEAD_TYPE="linear")

    # 2) k-NN readout (often strong too)
    ssl_handler.evaluate_model(ckpt_path="ssl-final.pt", HEAD_TYPE="knn", KNN_K=200, KNN_T=0.07)

    # 3) your original prototypes, but you can crank k per class
    ssl_handler.evaluate_model(ckpt_path="ssl-final.pt", HEAD_TYPE="proto", PROTOS_PER_CLASS=40, MARGIN_TH=0.0)
    #ssl_handler.evaluate_model()

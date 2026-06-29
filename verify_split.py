import numpy as np, torch, random
from torchvision import datasets
from dataloader import CIFAR10BYOLClientData

# Build a bare instance (bypass heavy __init__) and call the REAL _partition_indices.
ds = datasets.CIFAR10(root="./data", train=True, download=True)
targets = np.asarray(ds.targets)
n = len(targets)

K, L, SEED = 5, 2, 12345
obj = CIFAR10BYOLClientData.__new__(CIFAR10BYOLClientData)
obj.non_iid = True
obj.classes_per_client = L
obj.num_clients = K
obj.seed = SEED
obj._full_train_targets = targets

all_idx = []
print(f"{'cid':>3} | {'total':>6} | per-class counts (class:count)")
print("-"*70)
for cid in range(K):
    idx = obj._partition_indices(n, K, cid)
    all_idx.append(set(idx))
    labs = targets[idx]
    counts = {int(c): int((labs==c).sum()) for c in np.unique(labs)}
    nz = {c:v for c,v in counts.items() if v>0}
    print(f"{cid:>3} | {len(idx):>6} | {nz}")

# Overlap + coverage checks
print("-"*70)
union = set().union(*all_idx)
total = sum(len(s) for s in all_idx)
overlap = total - len(union)
print(f"sum of client sizes : {total}")
print(f"unique indices used : {len(union)}  (of {n} train images)")
print(f"duplicated indices  : {overlap}")
# class -> which clients hold it
cls_owner = {}
for cid in range(K):
    for c in np.unique(targets[list(all_idx[cid])]):
        cls_owner.setdefault(int(c), []).append(cid)
print(f"class -> clients    : {cls_owner}")
shared = {c:o for c,o in cls_owner.items() if len(o)>1}
print(f"classes shared by >1 client: {shared if shared else 'NONE (fully disjoint)'}")

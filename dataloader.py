import random
from typing import Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms as T
import matplotlib.pyplot as plt
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import ConcatDataset
class BYOLPairDataset(Dataset):
    """
    Wraps a base dataset and returns two augmented views per sample.
    Optionally keeps the original label.
    """
    def __init__(self, base_dataset: Dataset, transform, keep_labels: bool = False):
        self.base_dataset = base_dataset
        self.transform = transform
        self.keep_labels = keep_labels

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        v1 = self.transform(img)
        v2 = self.transform(img)
        if self.keep_labels:
            return (v1, v2), label
        else:
            return v1, v2


class CIFAR10BYOLClientData:
    """
    Object to be used by each FLWR client as its dataloader constructor.

    - Uses a fixed global seed so all clients see the same initial shuffle.
    - Splits CIFAR-10 train and test sets into `num_clients` disjoint slices.
    - For client `cid`, returns BYOL-style DataLoaders that yield (view1, view2)
      or ((view1, view2), label) if keep_labels=True.
    """

    def __init__(
        self,
        num_clients: int,
        classes_per_client: int,
        cid: int,
        batch_size: int = 128,
        data_dir: str = "./data",
        keep_labels: bool = False,
        num_workers: int = 2,
        seed: int = 12345,
        device: str = "cpu"
    ):
        """
        Args:
            num_clients: total number of FL clients.
            cid: integer client id in [0, num_clients-1].
            batch_size: batch size for both train and val loaders.
            data_dir: where to store CIFAR-10.
            keep_labels: if True, dataloaders also return labels.
            num_workers: DataLoader num_workers.
            seed: global seed for shuffling/partitioning (same for all clients).
        """
        assert 0 <= cid < num_clients, "cid must be in [0, num_clients-1]"
        self.non_iid = True
        self.classes_per_client = classes_per_client
        self.num_clients = num_clients
        self.cid = cid
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.keep_labels = keep_labels
        self.num_workers = num_workers
        self.seed = seed

        # Set global seeds for reproducibility of the *partitioning*
        random.seed(seed)
        torch.manual_seed(seed)

        # Define BYOL-style transforms for CIFAR-10
        self.train_transform = self._build_byol_transform(train=True)
        self.val_transform = self._build_byol_transform(train=False)

        # Load datasets
        self._prepare_datasets()
        #Decide on pinning memory
        if device=="cuda":
            self.pin_memory=True
        else:
            self.pin_memory=False
        # Create dataloaders for this client
        self.train_loader, self.val_loader = self._create_dataloaders()

    def _build_byol_transform(self, train: bool):
        """
        BYOL-ish transforms adapted for CIFAR-10 (32x32).
        """
        color_jitter = T.ColorJitter(0.8, 0.8, 0.8, 0.2)

        if train:
            #return T.Compose([
            #    T.RandomResizedCrop(32, scale=(0.2, 1.0)), 
            #    T.RandomHorizontalFlip(),
            #    T.RandomApply([color_jitter], p=0.8),
            #    T.RandomGrayscale(p=0.2),
            #    T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            #    T.ToTensor(),
            #    T.Normalize(
            #        mean=(0.4914, 0.4822, 0.4465),
            #        std=(0.2470, 0.2435, 0.2616),
            #    ),
            #])
            return T.Compose([
            #T.RandomResizedCrop(32, scale=(0.08, 1.0)),  # Stronger crop per paper/repo
            #T.RandomHorizontalFlip(),
            #T.RandomApply([color_jitter], p=0.8),
            #T.RandomGrayscale(p=0.2),
            #T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),  # Add blur, repo p=0.5
            #T.ToTensor(),
            #T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),  # Critical: add normalize

            torchvision.transforms.RandomResizedCrop(32, scale=(0.2, 1.)),#torchvision.transforms.RandomResizedCrop(size=32),
            torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
            torchvision.transforms.RandomApply([color_jitter], p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
            #T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))]),
            torchvision.transforms.ToTensor(),
            T.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616),
                ),
        ])

        else:
            # For "validation" you may want slightly weaker transforms,
            # but still produce two views for BYOL-style evaluation.
            return T.Compose([
                T.Resize(32),
                T.CenterCrop(32), #Potentially not needed
                T.ToTensor(),
                T.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616),
                ),
            ])
    def _partition_indices(self, n: int, num_clients: int, cid: int, 
                        generator: Optional[torch.Generator] = None):
        """
        EXACT paper non-IID partitioning (label heterogeneity with shards).
        - Splits each class into ceil((K*l)/C) shards.
        - Global shuffle of all shards.
        - Sequential assignment of l shards per client (distinct classes w/ high prob).
        - No data duplication; balanced volume (~n/K per client).
        - For l=2 (your case): full classes, disjoint.
        - For IID: set classes_per_client=10 (or non_iid=False below).
        """
        if not self.non_iid or self.classes_per_client == 10:  # ← Add this for clean IID
            # IID: random disjoint partition (paper's "all classes" split)
            if generator is None:
                generator = torch.Generator().manual_seed(self.seed)
            perm = torch.randperm(n, generator=generator)
            shard_size = n // num_clients
            start = cid * shard_size
            end = (cid + 1) * shard_size if cid < num_clients - 1 else n
            return perm[start:end].tolist()

        # ── Non-IID: paper's shards method ──
        labels = np.asarray(self._full_train_targets)  # (n,)
        num_classes = len(np.unique(labels))

        # Total shards/sets needed
        total_sets = num_clients * self.classes_per_client

        # Shards per class (key fix: NO max(2,); paper uses equal split)
        shards_per_class = int(np.ceil(total_sets / num_classes))

        # Build shards per class (shuffled within class for variety)
        all_shards = []  # list of index lists (one shard = pure-class indices)
        rng = np.random.default_rng(self.seed)  # reproducible

        for cls in range(num_classes):
            cls_idx = np.where(labels == cls)[0]
            rng.shuffle(cls_idx)  # per-class shuffle
            shard_size = len(cls_idx) // shards_per_class
            remainder = len(cls_idx) % shards_per_class
            pos = 0
            for i in range(shards_per_class):
                extra = 1 if i < remainder else 0
                shard_end = pos + shard_size + extra
                shard = cls_idx[pos:shard_end].tolist()
                all_shards.append(shard)
                pos = shard_end

        # Global shuffle of ALL shards (paper's "random assign")
        rng = np.random.default_rng(self.seed)
        rng.shuffle(all_shards)

        # Assign l shards to this client
        shards_per_client = self.classes_per_client
        start_shard = cid * shards_per_client
        end_shard = start_shard + shards_per_client
        client_shards = all_shards[start_shard:end_shard]

        # Flatten + per-client shuffle (standard)
        client_indices = []
        for shard in client_shards:
            client_indices.extend(shard)
        random.seed(self.seed + cid)
        random.shuffle(client_indices)

        return client_indices
    @torch.no_grad()  # if you need no-grad in some contexts – otherwise remove
    def _partition_indices_2(self, n: int, num_clients: int, cid: int, 
                        generator: Optional[torch.Generator] = None):
        """
        Partition indices for federated learning.
        
        - If not self.non_iid:      classic random IID split (disjoint shards)
        - If self.non_iid:          pathological non-IID (shards method)
        → each class split into multiple shards
        → each client gets exactly self.classes_per_client random shards
        → aims for roughly balanced size per client
        """
        if not self.non_iid:
            # ── IID: random disjoint partition ──
            if generator is None:
                generator = torch.Generator().manual_seed(self.seed)
            perm = torch.randperm(n, generator=generator)
            shard_size = n // num_clients
            start = cid * shard_size
            end = (cid + 1) * shard_size if cid < num_clients - 1 else n
            return perm[start:end].tolist()

        # ── Non-IID: shards-based pathological partitioning ──
        # Assume self._full_train_targets exists and contains all labels (list or np.array)
        labels = np.asarray(self._full_train_targets)   # shape (n,)
        num_classes = len(np.unique(labels))            # 10 for CIFAR-10

        # ── Decide number of shards per class ──
        # Follows common practice: roughly (K × l) total shards
        total_shards = num_clients * self.classes_per_client
        shards_per_class = max(2, int(np.ceil(total_shards / num_classes)))

        # For reproducibility across all clients
        rng = np.random.default_rng(self.seed)
        random.seed(self.seed)

        # ── 1. Collect all shards ──
        all_shards = []  # list of lists: pure-class index lists

        for cls in range(num_classes):
            cls_idx = np.where(labels == cls)[0]
            rng.shuffle(cls_idx)

            shard_size_base = len(cls_idx) // shards_per_class
            remainder = len(cls_idx) % shards_per_class

            pos = 0
            for i in range(shards_per_class):
                extra = 1 if i < remainder else 0
                shard_end = pos + shard_size_base + extra
                if pos < shard_end:  # avoid empty shards
                    all_shards.append(cls_idx[pos:shard_end].tolist())
                pos = shard_end

        # ── 2. Shuffle shards globally and assign l to each client ──
        rng.shuffle(all_shards)  # global random order

        # Each client gets self.classes_per_client shards
        shards_per_client = self.classes_per_client

        start_shard = cid * shards_per_client
        end_shard = start_shard + shards_per_client

        client_shards = all_shards[start_shard:end_shard]

        # Flatten + shuffle inside client (common & helps randomness)
        client_indices = []
        for shard in client_shards:
            client_indices.extend(shard)

        random.shuffle(client_indices)  # per-client shuffle, seeded above

        return client_indices

    def _partition_indices_old(self, n: int, num_clients: int, cid: int, generator: Optional[torch.Generator] = None):
        """
        Deterministically partition indices.
        - If self.non_iid: disjoint l classes per client (paper-style for CIFAR-10 l=2)
        - Else: your original random IID partitioning
        """
        if not self.non_iid:
            # Your original IID code (unchanged)
            if generator is None:
                generator = torch.Generator()
                generator.manual_seed(self.seed)

            perm = torch.randperm(n, generator=generator)
            shard_size = n // num_clients
            start = cid * shard_size
            end = (cid + 1) * shard_size if cid != num_clients - 1 else n
            idxs = perm[start:end]
            return idxs.tolist()

        # ── Non-IID: paper's l=2 classes per client ──
        # Load targets from the base dataset
        # (We need labels → assume base_dataset is CIFAR10 with .targets)
        targets = torch.tensor(self._full_train_targets)

        num_classes = 10  # CIFAR-10 hard-coded
        assigned_classes = []

        # Assign disjoint sets of classes_per_client classes to each client
        # Example for K=5, l=2: client 0 → classes 0,1; client 1 → 2,3; ... client 4 → 8,9
        start_class = cid * self.classes_per_client
        for i in range(self.classes_per_client):
            cls = (start_class + i) % num_classes  # modulo in case K*l > 10 (rare)
            assigned_classes.append(cls)

        idxs = []
        for cls in assigned_classes:
            class_indices = torch.nonzero(targets == cls).squeeze().tolist()
            idxs.extend(class_indices)

        # Shuffle within the client's data (good practice, paper likely does implicitly)
        random.seed(self.seed + cid)  # deterministic per client
        random.shuffle(idxs)

        return idxs

    def _prepare_datasets(self):
        """
        Loads CIFAR-10 train and test and creates client-specific subsets.
        """
        # Load full datasets (same on every client)
        full_train = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=None,  # transforms applied later in BYOLPairDataset
        )
        full_test = datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=None,
        )
        self._full_train_targets = full_train.targets
        self._full_test_targets  = full_test.targets
        # Create deterministic generators for train/test partitioning
        gen_train = torch.Generator().manual_seed(self.seed)
        gen_test = torch.Generator().manual_seed(self.seed + 1)  # different, but still deterministic

        # Partition indices
        train_indices = self._partition_indices(len(full_train), self.num_clients, self.cid, generator=gen_train)
        val_indices = self._partition_indices(len(full_test), self.num_clients, self.cid, generator=gen_test)

        # Client-specific subsets
        client_train_subset = Subset(full_train, train_indices)
        client_val_subset = Subset(full_test, val_indices)

        # Wrap with BYOL pair dataset
        self.client_train_byol = BYOLPairDataset(
            base_dataset=client_train_subset,
            transform=self.train_transform,
            keep_labels=self.keep_labels,
        )

        self.client_val_byol = BYOLPairDataset(
            base_dataset=client_val_subset,
            transform=self.val_transform,
            keep_labels=self.keep_labels,
        )

    def _create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and validation dataloaders for this client.
        """
        train_loader = DataLoader(
            self.client_train_byol,
            batch_size=self.batch_size,
            shuffle=True,            # local shuffle within the client's shard
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=self.pin_memory,
            drop_last=True,
            prefetch_factor=4,
        )

        val_loader = DataLoader(
            self.client_val_byol,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

        return train_loader, val_loader

    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Public method: returns (train_loader, val_loader)
        """
        return self.train_loader, self.val_loader

def build_eval_loaders(data_dir="./data", batch_size=512, num_workers=2):
    tfm = T.Compose([
        T.Resize(32),
        T.CenterCrop(32),
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616)),
    ])
    train_ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=tfm)
    test_ds  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=tfm)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_ld  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_ld, test_ld


def partition_indices_like_clients(
    targets,
    n: int,
    num_clients: int,
    cid: int,
    classes_per_client: int,
    seed: int = 12345,
    non_iid: bool = True,
):
    """
    Standalone version of CIFAR10BYOLClientData._partition_indices so the server can reuse it.
    Mirrors the logic in CIFAR10BYOLClientData._partition_indices.  [2](https://aaudk-my.sharepoint.com/personal/st28xu_es_aau_dk/Documents/Microsoft%20Copilot%20Chat%20Files/dataloader.py)
    """
    if (not non_iid) or (classes_per_client == 10):
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=gen)
        shard_size = n // num_clients
        start = cid * shard_size
        end = (cid + 1) * shard_size if cid < num_clients - 1 else n
        return perm[start:end].tolist()

    labels = np.asarray(targets)
    num_classes = len(np.unique(labels))

    total_sets = num_clients * classes_per_client
    shards_per_class = int(np.ceil(total_sets / num_classes))

    all_shards = []
    rng = np.random.default_rng(seed)
    for cls in range(num_classes):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        shard_size = len(cls_idx) // shards_per_class
        remainder = len(cls_idx) % shards_per_class
        pos = 0
        for i in range(shards_per_class):
            extra = 1 if i < remainder else 0
            shard_end = pos + shard_size + extra
            shard = cls_idx[pos:shard_end].tolist()
            all_shards.append(shard)
            pos = shard_end

    rng = np.random.default_rng(seed)
    rng.shuffle(all_shards)

    start_shard = cid * classes_per_client
    end_shard = start_shard + classes_per_client
    client_shards = all_shards[start_shard:end_shard]

    client_indices = []
    for shard in client_shards:
        client_indices.extend(shard)

    random.seed(seed + cid)
    random.shuffle(client_indices)
    return client_indices


def build_server_eval_loaders(
    data_dir: str = "./data",
    batch_size: int = 512,
    num_workers: int = 2,
    # server split config
    num_clients: int = 10,
    server_cid: int = 0,
    classes_per_client: int = 10,
    seed: int = 12345,
    non_iid: bool = False,
    # label budget
    labeled_per_class: int = 10,
    # what to evaluate on
    eval_on_remaining_train_plus_test: bool = True,
):
    """
    Returns:
      label_ld: loader over 100 labeled examples (10 per class) drawn from server split.
      eval_ld: loader over "the rest of CIFAR-10" (either test only, or remaining train + test).
    """
    tfm = T.Compose([
        T.Resize(32),
        T.CenterCrop(32),
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616)),
    ])

    full_train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=tfm)
    full_test  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=tfm)

    # Server "own split" indices (same idea as clients)
    server_split_idx = partition_indices_like_clients(
        targets=full_train.targets,
        n=len(full_train),
        num_clients=num_clients,
        cid=server_cid,
        classes_per_client=classes_per_client,
        seed=seed,
        non_iid=non_iid,
    )

    # Select exactly labeled_per_class per class from server split
    rng = np.random.default_rng(seed + 999)  # deterministic
    by_class = {c: [] for c in range(10)}
    for idx in server_split_idx:
        y = full_train.targets[idx]
        by_class[y].append(idx)

    chosen = []
    missing = []
    for c in range(10):
        if len(by_class[c]) < labeled_per_class:
            missing.append((c, len(by_class[c])))
            continue
        pick = rng.choice(by_class[c], size=labeled_per_class, replace=False)
        chosen.extend(pick.tolist())

    if missing:
        raise ValueError(
            "Server split does not contain enough samples for some classes. "
            f"Need {labeled_per_class}/class, but got: {missing}. "
            "Fix by setting non_iid=False or classes_per_client=10 for server evaluation."
        )

    chosen_set = set(chosen)

    label_ds = Subset(full_train, chosen)

    if eval_on_remaining_train_plus_test:
        remaining_train_idx = [i for i in range(len(full_train)) if i not in chosen_set]
        remaining_train_ds = Subset(full_train, remaining_train_idx)
        eval_ds = ConcatDataset([remaining_train_ds, full_test])
    else:
        eval_ds = full_test

    label_ld = DataLoader(label_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    eval_ld  = DataLoader(eval_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return label_ld, eval_ld




if __name__ == "__main__":
    data_obj = CIFAR10BYOLClientData(
        num_clients=101,
        cid=2,
        batch_size=128,
        keep_labels=False,
        data_dir="./data",
        seed=12345,  # must be the same on all clients
    )
    train_loader, val_loader = data_obj.get_loaders()
    v1, v2 = train_loader.dataset[0]

    def show(img):
        img = img.permute(1,2,0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())
        plt.imshow(img)
        plt.axis("off")

    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    show(v1)
    plt.title("View 1")

    plt.subplot(1,2,2)
    show(v2)
    plt.title("View 2")

    plt.show()
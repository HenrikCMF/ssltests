import random
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms as T
import matplotlib.pyplot as plt

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
        self.classes_per_client = 10
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
        color_jitter = T.ColorJitter(0.4, 0.4, 0.4, 0.1)

        if train:
            return T.Compose([
                T.RandomResizedCrop(32, scale=(0.2, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomApply([color_jitter], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                T.ToTensor(),
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
                T.CenterCrop(32),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616),
                ),
            ])

    def _partition_indices(self, n: int, num_clients: int, cid: int, generator: Optional[torch.Generator] = None):
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
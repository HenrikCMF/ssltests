import os
import random
from typing import Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms as T
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────── #
# Per-dataset normalization constants and asset URLs.
# ─────────────────────────────────────────────────────────────────────────── #
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD  = (0.2673, 0.2564, 0.2762)

TINY_IMAGENET_MEAN = (0.4802, 0.4481, 0.3975)
TINY_IMAGENET_STD  = (0.2302, 0.2265, 0.2262)
TINY_IMAGENET_URL  = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


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


def _stratified_keep_indices(targets, fraction: float, seed: int):
    """Return sorted indices that keep `fraction` of samples per class.

    Every class stays represented (at least one sample), only the count per
    class shrinks. Deterministic in `seed`. Works for any dataset given its
    per-sample class labels.
    """
    if fraction >= 1.0:
        return list(range(len(targets)))
    by_class: dict = {}
    for i, c in enumerate(targets):
        by_class.setdefault(int(c), []).append(i)
    rng = np.random.default_rng(seed)
    keep = []
    for cls in sorted(by_class):
        idx = np.asarray(by_class[cls])
        n_keep = max(1, int(round(len(idx) * fraction)))
        sel = rng.permutation(len(idx))[:n_keep]
        keep.extend(idx[sel].tolist())
    keep.sort()
    return keep


class BYOLClientData:
    """
    Base FLWR-client dataloader constructor shared by every dataset.

    Holds all of the logic common to CIFAR-10 / CIFAR-100 / Tiny ImageNet:
      - a fixed global seed so all clients see the same initial shuffle,
      - shard-based (paper) non-IID partitioning into `num_clients` slices,
      - BYOL-style transforms that yield (view1, view2) — or ((view1, view2),
        label) when keep_labels=True — with an optional ~2x "strong" mode,
      - train/val DataLoader construction and evaluation-loader helpers.

    A concrete dataset subclass only supplies the pieces that genuinely differ:
      - class attributes MEAN/STD/IMG_SIZE/NUM_CLASSES/USE_BLUR/BLUR_KERNEL/
        DEFAULT_LABELED_PER_CLASS/TORCHVISION_DS,
      - _load_full_datasets() and _make_eval_datasets() (how raw data is read),
      - optionally _maybe_setup() (one-time download/restructure) and
        _val_partition_targets() (which labels drive the val shard split).
    """

    # ── Dataset-specific knobs; overridden by each subclass ──────────────── #
    MEAN: Tuple[float, float, float] = None
    STD:  Tuple[float, float, float] = None
    IMG_SIZE: int = 32
    NUM_CLASSES: int = 10
    USE_BLUR: bool = False          # add GaussianBlur to the BYOL train views
    BLUR_KERNEL: int = 3
    DEFAULT_LABELED_PER_CLASS: int = 10  # server-eval labeled samples per class
    TORCHVISION_DS = None           # torchvision Dataset class (CIFAR-style loaders)

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
        device: str = "cpu",
        download: bool = True,
        data_fraction: float = 1.0,
        strong_aug: bool = False,
    ):
        """
        Args:
            num_clients: total number of FL clients.
            cid: integer client id in [0, num_clients-1].
            classes_per_client: non-IID shards (classes) assigned to each client.
            batch_size: batch size for both train and val loaders.
            data_dir: where to store / read the dataset.
            keep_labels: if True, dataloaders also return labels.
            num_workers: DataLoader num_workers.
            seed: global seed for shuffling/partitioning (same for all clients).
            download: when False, skip torchvision's ~390ms MD5 integrity check
                (files assumed present). Only the first construction per process
                needs download=True; see FedClient.
            data_fraction: fraction of training samples kept per class
                (1.0 = full dataset, 0.5 = half; all classes stay represented).
            strong_aug: when True, ~double the magnitude of every BYOL train
                augmentation (simulates clients that deliberately harden their
                views against reconstruction). See _build_byol_transform.
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
        self.data_fraction = data_fraction
        self.download = download
        self.strong_aug = strong_aug

        # Set global seeds for reproducibility of the *partitioning*.
        random.seed(seed)
        torch.manual_seed(seed)

        # BYOL-style transforms (two views come from applying train_transform twice).
        self.train_transform = self._build_byol_transform(train=True)
        self.val_transform = self._build_byol_transform(train=False)

        self._maybe_setup()          # subclass hook (e.g. TinyImageNet download)
        self._prepare_datasets()     # template method using subclass data hooks

        self.pin_memory = device == "cuda"
        self.train_loader, self.val_loader = self._create_dataloaders()

    # ------------------------------------------------------------------ #
    # Transforms
    # ------------------------------------------------------------------ #
    def _build_byol_transform(self, train: bool):
        """BYOL-ish transforms sized/normalized per the subclass's class attrs.

        When self.strong_aug is True, every augmentation is scaled up ~2x:
        crop min-scale 0.2->0.1, ColorJitter (0.8,0.8,0.8,0.2)->(1.6,1.6,1.6,0.4),
        grayscale p 0.2->0.4, and (where USE_BLUR) blur apply-p 0.5->1.0.
        Horizontal flip stays at p=0.5 (already symmetric / max-entropy).
        """
        if self.strong_aug:
            color_jitter = T.ColorJitter(1.6, 1.6, 1.6, 0.4)
            crop_min_scale, grayscale_p, blur_p = 0.1, 0.4, 1.0
        else:
            color_jitter = T.ColorJitter(0.8, 0.8, 0.8, 0.2)
            crop_min_scale, grayscale_p, blur_p = 0.2, 0.2, 0.5

        if train:
            ops = [
                T.RandomResizedCrop(self.IMG_SIZE, scale=(crop_min_scale, 1.0)),
                T.RandomHorizontalFlip(),  # with 0.5 probability
                T.RandomApply([color_jitter], p=0.8),
                T.RandomGrayscale(p=grayscale_p),
            ]
            if self.USE_BLUR:
                ops.append(
                    T.RandomApply(
                        [T.GaussianBlur(kernel_size=self.BLUR_KERNEL, sigma=(0.1, 2.0))],
                        p=blur_p,
                    )
                )
            ops += [T.ToTensor(), T.Normalize(mean=self.MEAN, std=self.STD)]
            return T.Compose(ops)

        # "Validation" view: deterministic resize/crop, still produces two views
        # for BYOL-style evaluation.
        return T.Compose([
            T.Resize(self.IMG_SIZE),
            T.CenterCrop(self.IMG_SIZE),
            T.ToTensor(),
            T.Normalize(mean=self.MEAN, std=self.STD),
        ])

    # ------------------------------------------------------------------ #
    # Non-IID shard partitioning (paper Alg.: label heterogeneity w/ shards)
    # ------------------------------------------------------------------ #
    def _shard_partition(self, targets, n: int,
                         generator: Optional[torch.Generator] = None):
        """
        EXACT paper non-IID partitioning (label heterogeneity with shards).
        - Splits each class into ceil((K*l)/C) shards.
        - Global shuffle of all shards.
        - Sequential assignment of l shards per client (distinct classes w/ high prob).
        - No data duplication; balanced volume (~n/K per client).
        - IID fast path when non_iid is off or classes_per_client == NUM_CLASSES.

        `targets` is the label array the shards are built from; `n` is only used
        by the IID fast path.
        """
        if (not self.non_iid) or (self.classes_per_client == self.NUM_CLASSES):
            # IID: random disjoint partition (paper's "all classes" split)
            if generator is None:
                generator = torch.Generator().manual_seed(self.seed)
            perm = torch.randperm(n, generator=generator)
            shard_size = n // self.num_clients
            start = self.cid * shard_size
            end = (self.cid + 1) * shard_size if self.cid < self.num_clients - 1 else n
            return perm[start:end].tolist()

        # ── Non-IID: paper's shards method ──
        labels = np.asarray(targets)
        num_classes = len(np.unique(labels))

        total_sets = self.num_clients * self.classes_per_client
        # Shards per class (paper uses equal split; NO max(2,)).
        shards_per_class = int(np.ceil(total_sets / num_classes))

        # Build shards per class (shuffled within class for variety).
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
                all_shards.append(cls_idx[pos:shard_end].tolist())
                pos = shard_end

        # Global shuffle of ALL shards (paper's "random assign").
        rng = np.random.default_rng(self.seed)
        rng.shuffle(all_shards)

        # Assign l shards to this client, flatten, then per-client shuffle.
        start_shard = self.cid * self.classes_per_client
        end_shard = start_shard + self.classes_per_client
        client_indices = [idx for shard in all_shards[start_shard:end_shard] for idx in shard]
        random.seed(self.seed + self.cid)
        random.shuffle(client_indices)
        return client_indices

    # ------------------------------------------------------------------ #
    # Dataset assembly (template method + subclass data hooks)
    # ------------------------------------------------------------------ #
    def _maybe_setup(self):
        """One-time download/restructure hook. No-op by default."""
        pass

    def _load_full_datasets(self):
        """Return (full_train, full_test) with transform=None, data_fraction applied.

        Default is the CIFAR-style torchvision loader driven by TORCHVISION_DS;
        datasets with a different on-disk layout (e.g. Tiny ImageNet) override.
        """
        full_train = self.TORCHVISION_DS(
            root=self.data_dir, train=True, download=self.download, transform=None,
        )
        full_test = self.TORCHVISION_DS(
            root=self.data_dir, train=False, download=self.download, transform=None,
        )
        if self.data_fraction < 1.0:
            keep = _stratified_keep_indices(full_train.targets, self.data_fraction, self.seed)
            full_train.data = full_train.data[keep]
            full_train.targets = [full_train.targets[i] for i in keep]
        return full_train, full_test

    def _val_partition_targets(self, full_train, full_test):
        """Label array used to shard the *val* split.

        CIFAR-10/100 historically partition the val split with the *train*
        labels (preserved here). TinyImageNet overrides to use val labels.
        """
        return self._full_train_targets

    def _prepare_datasets(self):
        full_train, full_test = self._load_full_datasets()
        self._full_train_targets = list(full_train.targets)
        self._full_test_targets = list(full_test.targets)

        # Deterministic-but-distinct generators for the train/val partitions.
        gen_train = torch.Generator().manual_seed(self.seed)
        gen_test = torch.Generator().manual_seed(self.seed + 1)

        train_indices = self._shard_partition(
            self._full_train_targets, len(full_train), generator=gen_train
        )
        val_indices = self._shard_partition(
            self._val_partition_targets(full_train, full_test),
            len(full_test), generator=gen_test,
        )

        self.client_train_byol = BYOLPairDataset(
            base_dataset=Subset(full_train, train_indices),
            transform=self.train_transform,
            keep_labels=self.keep_labels,
        )
        self.client_val_byol = BYOLPairDataset(
            base_dataset=Subset(full_test, val_indices),
            transform=self.val_transform,
            keep_labels=self.keep_labels,
        )

    def _create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(
            self.client_train_byol,
            batch_size=self.batch_size,
            shuffle=True,            # local shuffle within the client's shard
            num_workers=self.num_workers,
            persistent_workers=False,
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
        """Public method: returns (train_loader, val_loader)."""
        return self.train_loader, self.val_loader

    # ------------------------------------------------------------------ #
    # Evaluation loaders (class-level; no client partitioning)
    # ------------------------------------------------------------------ #
    @classmethod
    def _eval_transform(cls):
        return T.Compose([
            T.Resize(cls.IMG_SIZE),
            T.CenterCrop(cls.IMG_SIZE),
            T.ToTensor(),
            T.Normalize(mean=cls.MEAN, std=cls.STD),
        ])

    @classmethod
    def _make_eval_datasets(cls, data_dir, tfm):
        """Return (train_ds, test_ds) with `tfm` applied. CIFAR-style default."""
        train_ds = cls.TORCHVISION_DS(root=data_dir, train=True, download=True, transform=tfm)
        test_ds = cls.TORCHVISION_DS(root=data_dir, train=False, download=True, transform=tfm)
        return train_ds, test_ds

    @classmethod
    def build_eval_loaders(cls, data_dir="./data", batch_size=512, num_workers=2):
        tfm = cls._eval_transform()
        train_ds, test_ds = cls._make_eval_datasets(data_dir, tfm)
        train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=False)
        test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=False)
        return train_ld, test_ld

    @staticmethod
    def partition_indices_like_clients(
        targets,
        n: int,
        num_clients: int,
        cid: int,
        classes_per_client: int,
        seed: int = 12345,
        non_iid: bool = True,
    ):
        """Standalone replica of the client shard split (dataset-agnostic).

        Used by build_server_eval_loaders to reproduce a client's split without
        constructing a full client-data object.
        """
        if (not non_iid) or (classes_per_client == len(np.unique(np.asarray(targets)))):
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
                all_shards.append(cls_idx[pos:shard_end].tolist())
                pos = shard_end

        rng = np.random.default_rng(seed)
        rng.shuffle(all_shards)

        start_shard = cid * classes_per_client
        end_shard = start_shard + classes_per_client
        client_indices = [idx for shard in all_shards[start_shard:end_shard] for idx in shard]

        random.seed(seed + cid)
        random.shuffle(client_indices)
        return client_indices

    @classmethod
    def build_server_eval_loaders(
        cls,
        data_dir: str = "./data",
        batch_size: int = 512,
        num_workers: int = 2,
        num_clients: int = 10,
        server_cid: int = 0,
        classes_per_client: Optional[int] = None,
        seed: int = 12345,
        non_iid: bool = False,
        labeled_per_class: Optional[int] = None,
        eval_on_remaining_train_plus_test: bool = True,
    ):
        """
        Returns:
          label_ld: loader over labeled examples (labeled_per_class per class)
            drawn from the server split.
          eval_ld: loader over "the rest" (either test only, or remaining
            train + test).
        """
        if classes_per_client is None:
            classes_per_client = cls.NUM_CLASSES
        if labeled_per_class is None:
            labeled_per_class = cls.DEFAULT_LABELED_PER_CLASS

        tfm = cls._eval_transform()
        full_train, full_test = cls._make_eval_datasets(data_dir, tfm)
        targets = list(full_train.targets)

        server_split_idx = cls.partition_indices_like_clients(
            targets=targets,
            n=len(full_train),
            num_clients=num_clients,
            cid=server_cid,
            classes_per_client=classes_per_client,
            seed=seed,
            non_iid=non_iid,
        )

        rng = np.random.default_rng(seed + 999)
        by_class = {c: [] for c in range(cls.NUM_CLASSES)}
        for idx in server_split_idx:
            by_class[targets[idx]].append(idx)

        chosen, missing = [], []
        for c in range(cls.NUM_CLASSES):
            if len(by_class[c]) < labeled_per_class:
                missing.append((c, len(by_class[c])))
                continue
            pick = rng.choice(by_class[c], size=labeled_per_class, replace=False)
            chosen.extend(pick.tolist())

        if missing:
            raise ValueError(
                "Server split does not contain enough samples for some classes. "
                f"Need {labeled_per_class}/class, but got: {missing}. "
                f"Fix by setting non_iid=False or classes_per_client="
                f"{cls.NUM_CLASSES} for server evaluation."
            )

        chosen_set = set(chosen)
        label_ds = Subset(full_train, chosen)

        if eval_on_remaining_train_plus_test:
            remaining = [i for i in range(len(full_train)) if i not in chosen_set]
            eval_ds = ConcatDataset([Subset(full_train, remaining), full_test])
        else:
            eval_ds = full_test

        label_ld = DataLoader(label_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        eval_ld = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return label_ld, eval_ld


class CIFAR10BYOLClientData(BYOLClientData):
    """
    BYOL client data loader for CIFAR-10 (10 classes, 32x32 RGB).

    - Uses a fixed global seed so all clients see the same initial shuffle.
    - Splits CIFAR-10 train and test sets into `num_clients` disjoint slices.
    - For client `cid`, returns BYOL-style DataLoaders that yield (view1, view2)
      or ((view1, view2), label) if keep_labels=True.
    """

    MEAN = CIFAR10_MEAN
    STD = CIFAR10_STD
    IMG_SIZE = 32
    NUM_CLASSES = 10
    DEFAULT_LABELED_PER_CLASS = 10
    TORCHVISION_DS = datasets.CIFAR10


class CIFAR100BYOLClientData(BYOLClientData):
    """
    BYOL client data loader for CIFAR-100 (100 classes, 32x32 RGB).

    Mirrors CIFAR10BYOLClientData exactly (same shard-based non-IID partitioning,
    same BYOL transforms) but for the 100-class dataset. Auto-downloaded via
    torchvision. The IID special-case triggers when classes_per_client equals the
    number of classes (100), matching CIFAR-10's `== 10` behaviour.
    """

    MEAN = CIFAR100_MEAN
    STD = CIFAR100_STD
    IMG_SIZE = 32
    NUM_CLASSES = 100
    DEFAULT_LABELED_PER_CLASS = 10
    TORCHVISION_DS = datasets.CIFAR100


class TinyImageNetBYOLClientData(BYOLClientData):
    """
    BYOL client data loader for Tiny ImageNet (200 classes, 64x64 RGB).
    Downloads and sets up the dataset on first use.
    """

    MEAN = TINY_IMAGENET_MEAN
    STD = TINY_IMAGENET_STD
    IMG_SIZE = 64
    NUM_CLASSES = 200
    USE_BLUR = True
    BLUR_KERNEL = 5
    DEFAULT_LABELED_PER_CLASS = 5
    TORCHVISION_DS = None  # uses ImageFolder instead (see hooks below)

    def _maybe_setup(self):
        if self.download:
            self._setup(self.data_dir)

    @staticmethod
    def _setup(data_dir: str):
        """Download and restructure Tiny ImageNet val set if not already done.

        Uses val_annotations.txt presence as the sentinel: once restructuring
        completes the file is deleted, so this is safe to call multiple times.
        """
        import zipfile, urllib.request, shutil
        root = os.path.join(data_dir, "tiny-imagenet-200")
        zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")

        if not os.path.isdir(root):
            if not os.path.isfile(zip_path):
                print("Downloading Tiny ImageNet (~237 MB)...")
                urllib.request.urlretrieve(TINY_IMAGENET_URL, zip_path)
            print("Extracting...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(data_dir)

        val_dir = os.path.join(root, "val")
        ann_file = os.path.join(val_dir, "val_annotations.txt")
        if not os.path.isfile(ann_file):
            return  # already restructured

        print("Restructuring val set...")
        img_src_dir = os.path.join(val_dir, "images")
        with open(ann_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                fname, wnid = parts[0], parts[1]
                dst_dir = os.path.join(val_dir, wnid)
                os.makedirs(dst_dir, exist_ok=True)
                dst = os.path.join(dst_dir, fname)
                # Handle both a fresh extract (val/images/) and a previous bad
                # run that put files one level too deep (val/<wnid>/images/).
                for src in [
                    os.path.join(img_src_dir, fname),
                    os.path.join(val_dir, wnid, "images", fname),
                ]:
                    if os.path.isfile(src):
                        os.rename(src, dst)
                        break

        # Remove the now-empty flat images dir and any leftover <wnid>/images/ subdirs.
        if os.path.isdir(img_src_dir):
            shutil.rmtree(img_src_dir)
        for entry in os.scandir(val_dir):
            if entry.is_dir():
                nested = os.path.join(entry.path, "images")
                if os.path.isdir(nested):
                    shutil.rmtree(nested)

        os.remove(ann_file)
        print("Tiny ImageNet ready.")

    def _load_full_datasets(self):
        root = os.path.join(self.data_dir, "tiny-imagenet-200")
        full_train = datasets.ImageFolder(os.path.join(root, "train"))
        full_test = datasets.ImageFolder(os.path.join(root, "val"))

        if self.data_fraction < 1.0:
            keep = _stratified_keep_indices(
                [s[1] for s in full_train.samples], self.data_fraction, self.seed
            )
            full_train.samples = [full_train.samples[i] for i in keep]
            full_train.targets = [s[1] for s in full_train.samples]
            full_train.imgs = full_train.samples
        return full_train, full_test

    def _val_partition_targets(self, full_train, full_test):
        # TinyImageNet shards its val split by *val* labels (not train labels).
        return self._full_test_targets

    @classmethod
    def _make_eval_datasets(cls, data_dir, tfm):
        cls._setup(data_dir)
        root = os.path.join(data_dir, "tiny-imagenet-200")
        train_ds = datasets.ImageFolder(os.path.join(root, "train"), transform=tfm)
        test_ds = datasets.ImageFolder(os.path.join(root, "val"), transform=tfm)
        return train_ds, test_ds


if __name__ == "__main__":
    data_obj = CIFAR10BYOLClientData(
        num_clients=101,
        classes_per_client=2,
        cid=2,
        batch_size=128,
        keep_labels=False,
        data_dir="./data",
        seed=12345,  # must be the same on all clients
    )
    train_loader, val_loader = data_obj.get_loaders()
    v1, v2 = train_loader.dataset[0]

    def show(img):
        img = img.permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())
        plt.imshow(img)
        plt.axis("off")

    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    show(v1)
    plt.title("View 1")

    plt.subplot(1, 2, 2)
    show(v2)
    plt.title("View 2")

    plt.show()

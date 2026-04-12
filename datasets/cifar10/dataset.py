import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision import transforms


class CIFAR10Dataset(Dataset):
    def __init__(self, root, split="train", val_frac=0.1, seed=42, download=True):
        super().__init__()
        assert split in {"train", "val", "test"}
        self.split = split

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform_eval = transforms.Compose([
            transforms.ToTensor(),
        ])

        if split == "test":
            ds = CIFAR10(root=root, train=False, download=download, transform=transform_eval)
            self.base_dataset = ds
            self.indices = None
            return

        # one view with augmentation for train and
        # one deterministic view for val splitting / evaluation
        ds_train_aug = CIFAR10(root=root, train=True, download=download, transform=transform_train)
        ds_train_eval = CIFAR10(root=root, train=True, download=download, transform=transform_eval)

        targets = torch.tensor(ds_train_eval.targets, dtype=torch.long)
        n = len(targets)

        if val_frac is None or val_frac <= 0.0:
            if split == "val":
                raise RuntimeError("Validation requested but val_frac <= 0 produced no val set.")
            self.base_dataset = ds_train_aug
            self.indices = list(range(n))
            return

        g = torch.Generator()
        g.manual_seed(int(seed))

        train_idx = []
        val_idx = []

        # stratified split by class
        classes = torch.unique(targets).tolist()
        for c in classes:
            cls_idx = torch.nonzero(targets == c, as_tuple=False).view(-1)
            perm = cls_idx[torch.randperm(len(cls_idx), generator=g)]
            n_val = int(round(len(perm) * float(val_frac)))
            val_idx.extend(perm[:n_val].tolist())
            train_idx.extend(perm[n_val:].tolist())

        train_idx = sorted(train_idx)
        val_idx = sorted(val_idx)

        if split == "train":
            self.base_dataset = ds_train_aug
            self.indices = train_idx
        else:
            self.base_dataset = ds_train_eval
            self.indices = val_idx


    def __len__(self):
        if self.indices is None:
            return len(self.base_dataset)
        return len(self.indices)


    def __getitem__(self, idx):
        if self.indices is None:
            x, y = self.base_dataset[idx]
        else:
            x, y = self.base_dataset[self.indices[idx]]
        return x, y
    

    def get_targets(self):
        labels = self.base_dataset.targets
        if self.indices is not None:
            labels = [labels[i] for i in self.indices]
        return torch.as_tensor(labels, dtype=torch.long)
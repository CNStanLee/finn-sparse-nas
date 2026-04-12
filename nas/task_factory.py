import torch, random
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Sampler

from models.brevitas_mlp import JetSubstructureModel
from datasets.jsc_logicnets.dataset import JetSubstructureDataset

from models.brevitas_cnv import CIFAR10Model
from datasets.cifar10.dataset import CIFAR10Dataset


class SqrHingeLoss(nn.Module):
    def forward(self, logits, targets_pm1):
        return torch.mean(torch.clamp(1.0 - logits * targets_pm1, min=0.0) ** 2)
    

class StratifiedRollingSampler(Sampler):
    """
    Each __iter__ returns a class-balanced subset of size sum(per_class over classes).
    Across epochs, it rotates through each class index list with wraparound.
    """

    def __init__(self, labels, per_class, seed=42, shuffle_within_epoch=True):
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.per_class = int(per_class)
        self.seed = int(seed)
        self.shuffle_within_epoch = bool(shuffle_within_epoch)
        self.epoch = 0
        self.classes = sorted(torch.unique(self.labels).tolist())
        self.class_indices = {}
        g = torch.Generator()
        g.manual_seed(self.seed)
        for c in self.classes:
            idx = torch.nonzero(self.labels == c, as_tuple=False).view(-1)
            perm = idx[torch.randperm(len(idx), generator=g)]
            self.class_indices[c] = perm.tolist()
        self.num_samples = sum(min(self.per_class, len(self.class_indices[c])) for c in self.classes)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        epoch_indices = []
        for c in self.classes:
            cls_idx = self.class_indices[c]
            n = len(cls_idx)
            k = min(self.per_class, n)
            if n == 0 or k == 0:
                continue
            start = (self.epoch * self.per_class) % n
            chosen = [cls_idx[(start + i) % n] for i in range(k)]
            epoch_indices.extend(chosen)
        if self.shuffle_within_epoch:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(epoch_indices)
        self.epoch += 1
        return iter(epoch_indices)


def make_fixed_stratified_subset(ds, per_class, seed):
    if per_class is None or per_class <= 0:
        return ds
    labels = ds.get_targets()
    classes = torch.unique(labels).tolist()
    g = torch.Generator()
    g.manual_seed(int(seed))
    keep = []
    for c in classes:
        cls_idx = torch.nonzero(labels == c, as_tuple=False).view(-1)
        perm = cls_idx[torch.randperm(len(cls_idx), generator=g)]
        take = min(per_class, len(perm))
        keep.extend(perm[:take].tolist())
    keep = sorted(keep)
    return Subset(ds, keep)


def make_dataloaders(cfg, phase="quick"):
    task_name = cfg["task"]["name"]
    tcfg = cfg["train"][phase]
    if task_name == "jsc_mlp":
        ds_tr = JetSubstructureDataset(cfg["data"]["input_file"], cfg["data"]["config_file"], split="train", val_frac=cfg["data"]["val_frac"], seed=cfg["data"]["seed"])
        ds_va = JetSubstructureDataset(cfg["data"]["input_file"], cfg["data"]["config_file"], split="val",   val_frac=cfg["data"]["val_frac"], seed=cfg["data"]["seed"])
        dl_tr = DataLoader(ds_tr, batch_size=tcfg["batch_size"], shuffle=True, drop_last=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
        dl_va = DataLoader(ds_va, batch_size=tcfg["val_batch_size"], shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
        return dl_tr, dl_va
    elif task_name == "cifar10_cnv":
        ds_tr = CIFAR10Dataset(root=cfg["data"]["root"], split="train", val_frac=cfg["data"]["val_frac"], seed=cfg["data"]["seed"], download=cfg["data"]["download"])
        ds_va = CIFAR10Dataset(root=cfg["data"]["root"], split="val", val_frac=cfg["data"]["val_frac"], seed=cfg["data"]["seed"], download=cfg["data"]["download"])
        if phase == "quick":
            train_sampler = StratifiedRollingSampler(ds_tr.get_targets(), per_class=tcfg["train_subset_per_class"], seed=cfg["data"]["seed"], shuffle_within_epoch=True)
            ds_va = make_fixed_stratified_subset(ds_va, tcfg["val_subset_per_class"], cfg["data"]["seed"] + 1)
            dl_tr = DataLoader(ds_tr, batch_size=tcfg["batch_size"], shuffle=False, sampler=train_sampler, drop_last=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
        else:
            dl_tr = DataLoader(ds_tr, batch_size=tcfg["batch_size"], shuffle=True, drop_last=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
        dl_va = DataLoader(ds_va, batch_size=tcfg["val_batch_size"], shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
        return dl_tr, dl_va
    else:
        raise ValueError(f"Unsupported task name: {task_name}")


def make_test_dataloader(cfg):
    task_name = cfg["task"]["name"]
    tcfg = cfg["train"]["full"]
    if task_name == "jsc_mlp":
        ds_te = JetSubstructureDataset(cfg["data"]["input_file"], cfg["data"]["config_file"], split="test", val_frac=cfg["data"]["val_frac"], seed=cfg["data"]["seed"])
    elif task_name == "cifar10_cnv":
        ds_te = CIFAR10Dataset(root=cfg["data"]["root"], split="test", val_frac=cfg["data"]["val_frac"], seed=cfg["data"]["seed"], download=cfg["data"]["download"])
    else:
        raise ValueError(f"Unsupported task name: {task_name}")
    return DataLoader(ds_te, batch_size=tcfg["val_batch_size"], shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)


def build_model(cfg, cand):
    task_name = cfg["task"]["name"]
    q = cand["quant"]
    if task_name == "jsc_mlp":
        return JetSubstructureModel(
            cfg["task"]["in_features"],
            cfg["task"]["n_classes"],
            tuple(cand["hidden"]), 
            q["WB"], q["IA"], q["HA"], q["OA"]
        )
    elif task_name == "cifar10_cnv":
        d = cfg["model"]["defaults"]
        return CIFAR10Model(
            cfg["task"]["n_classes"],
            cfg["task"]["in_channels"],
            q["WB"], q["AB"], d["in_bits"],
            tuple(cand["conv_channels"]),
            tuple(cand["fc_features"]),
            tuple(d["pool_after"]),
            d["kernel_size"],
            cfg["task"]["img_h"],
            cfg["task"]["img_w"]
        )
    else:
        raise ValueError(f"Unsupported task name: {task_name}")


def make_training_recipe(cfg, model, phase="quick"):
    task_name = cfg["task"]["name"]
    tcfg = cfg["train"][phase]
    if task_name == "jsc_mlp":
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg.get("weight_decay", 0.0)))
        criterion = nn.CrossEntropyLoss()
        scheduler = None
    elif task_name == "cifar10_cnv":
        optimizer = torch.optim.Adam(model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg.get("weight_decay", 0.0)))
        criterion = SqrHingeLoss()
        scheduler = None
        if "lr_decay_every" in tcfg and "lr_decay_factor" in tcfg:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(tcfg["lr_decay_every"]), gamma=float(tcfg["lr_decay_factor"]))
    else:
        raise ValueError(f"Unsupported task name: {task_name}")
    return {
        "optimizer": optimizer,
        "criterion": criterion,
        "scheduler": scheduler,
        "epochs": int(tcfg["epochs"]),
        "early_stop_patience": tcfg.get("early_stop_patience"),
        "min_delta": float(tcfg.get("min_delta", 0.0)),
        "min_epochs": int(tcfg.get("min_epochs", 0)),
    }


def compute_loss(cfg, recipe, logits, yb, device):
    task_name = cfg["task"]["name"]
    if task_name == "jsc_mlp":
        return recipe["criterion"](logits, yb)
    elif task_name == "cifar10_cnv":
        n_classes = int(cfg["task"]["n_classes"])
        targets_pm1 = torch.full((yb.size(0), n_classes), -1.0, device=device)
        targets_pm1.scatter_(1, yb.unsqueeze(1), 1.0)
        return recipe["criterion"](logits, targets_pm1)
    else:
        raise ValueError(f"Unsupported task name: {task_name}")


def post_opt_step(cfg, model):
    if cfg["task"]["name"] == "cifar10_cnv" and hasattr(model, "clip_weights"):
        model.clip_weights(-1.0, 1.0)
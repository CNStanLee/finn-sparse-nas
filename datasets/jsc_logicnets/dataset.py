#  Copyright (C) 2021 Xilinx, Inc
#  Copyright (C) 2020 FastML
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


# Adapted from: Xilinx/logicnets/examples/jet_substructure/dataset.py
# Original authorship & license retained in header comments.
#
# Why adapted:
# - expose a validation split (train/val/test) with stratification
# - return integer class indices (not one-hot) for PyTorch CrossEntropyLoss
# - preserve column order; remove j_index by name; use yaml.safe_load

import h5py
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch
from torch.utils.data import Dataset


class JetSubstructureDataset(Dataset):
    def __init__(self, input_file, config_file, split="train", val_frac=0.1, seed=42):
        super().__init__()
        assert split in {"train", "val", "test"}
        self.split = split

        with h5py.File(input_file, 'r') as h5py_file:
            tree_array = h5py_file["t_allpar_new"][()]

        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)

        feature_labels = list(self.config["Inputs"])
        output_labels  = list(self.config["Labels"])

        if "j_index" in feature_labels:
            feature_labels = [c for c in feature_labels if c != "j_index"]
        if "j_index" in output_labels:
            output_labels = [c for c in output_labels if c != "j_index"]

        dataset_df = pd.DataFrame(tree_array, columns=(feature_labels+output_labels)).drop_duplicates()

        X_all = dataset_df[feature_labels].to_numpy()
        Y_all = dataset_df[output_labels].to_numpy() # one-hot

        # 1D class labels
        y_all = np.argmax(Y_all, axis=1).astype(np.int64)

        # 1st split: train_val vs test (80/20)
        X_tv, X_te, Y_tv, Y_te, y_tv, y_te = train_test_split(
            X_all, Y_all, y_all, test_size=0.2, random_state=42, stratify=y_all
        )

        # normalization
        if self.config.get("NormalizeInputs", True):
            scaler = preprocessing.StandardScaler().fit(X_tv)
            X_tv = scaler.transform(X_tv)
            X_te = scaler.transform(X_te)

        # PCA
        if self.config.get("ApplyPca", False):
            with torch.no_grad():
                dim = int(self.config.get("PcaDimensions", X_tv.shape[1]))
                X_tv_fp64 = torch.from_numpy(X_tv).double()
                X_te_fp64 = torch.from_numpy(X_te).double()
                U, S, V = torch.svd(X_tv_fp64)
                X_tv_pca_fp64 = torch.mm(X_tv_fp64, V[:, 0:dim])
                X_te_pca_fp64 = torch.mm(X_te_fp64, V[:, 0:dim])
                variance_retained = 100 * (S[0:dim].sum() / S.sum())
                print(f"Dimensions used for PCA: {dim}")
                print(f"Variance retained (%): {variance_retained}")
                X_tv = X_tv_pca_fp64.float().numpy()
                X_te = X_te_pca_fp64.float().numpy()

        # 2nd split: train vs val from the train_val pool
        if val_frac and val_frac > 0.0:
            X_tr, X_va, Y_tr, Y_va, y_tr, y_va = train_test_split(
                X_tv, Y_tv, y_tv, test_size=val_frac, random_state=seed, stratify=y_tv
            )
        else:
            X_tr, Y_tr, y_tr = X_tv, Y_tv, y_tv
            X_va = Y_va = y_va = None

        # final tensors: X as float32, y as class indices (int64)
        to_float = lambda a: torch.from_numpy(a.astype(np.float32))
        packs = {
            "train": (to_float(X_tr), torch.from_numpy(y_tr)),
            "val":   (to_float(X_va), torch.from_numpy(y_va)) if X_va is not None else (None, None),
            "test":  (to_float(X_te), torch.from_numpy(y_te)),
        }

        X_, y_ = packs[self.split]
        if X_ is None or y_ is None:
            raise RuntimeError("Validation requested but val_frac=0 produced no val set.")

        self.X, self.y = X_, y_


    def __len__(self):
        return len(self.X)


    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])

import os
import pickle

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset


# label encoder
def load_label_encoder(pkl_path, data):
    label_encoder = LabelEncoder()
    if not os.path.exists(pkl_path):
        label_encoder.fit(data)

        with open(pkl_path, "wb") as f:
            pickle.dump(label_encoder, f)

    else:
        with open(pkl_path, "rb") as f:
            label_encoder = pickle.load(f)

    return label_encoder


class NewsLogDataset(Dataset):
    def __init__(
        self,
        x_data=None,
        y_data=None,
        mode="train",
    ):
        self.mode = mode  ##train, valid, inference
        self.x_data = x_data if x_data is not None else None
        self.y_data = y_data if y_data is not None else None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.Tensor(self.x_data[idx]).to(self.device)
        if self.mode in ["train", "valid"]:
            y = torch.Tensor(self.y_data[idx]).to(self.device)

            return x, y
        elif self.mode == "inference":
            x = self.x[idx]
            return x


class NewsDataset(Dataset):
    def __init__(self, interaction=None, y=None, mode="train"):
        self.interaction = interaction
        self.y = y
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mode = mode

    def __len__(self):
        return len(self.interaction)

    def __getitem__(self, idx):
        interaction = torch.Tensor(self.interaction[idx]).to(self.device)

        if self.mode == "train":
            return interaction

        elif self.mode == "valid":
            actual = torch.tensor(self.y[idx])
            return interaction, actual

        elif self.mode == "inference":
            return interaction

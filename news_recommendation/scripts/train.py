import random
import numpy as np

import torch
import pandas as pd
from scipy.sparse import csr_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.optim as optim

from news_recommendation.scripts import preprocessing
from news_recommendation.models import bivae
from news_recommendation.utils import (
    train_test_split,
    dataset,
    early_stopping,
    augmentation,
    train,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(42)


train_loader = preprocessing.train_dataloader
eval_loader = preprocessing.test_dataloader

INPUT_DIM = len(train_loader.dataset[0])

model = bivae.VAE(hidden_dim=600, latent_dim=200, input_dim=INPUT_DIM)
optimizer = optim.Adam(params=model.parameters(), lr=1e-4)


# 72245 + 100000(0.0001)
EPOCH = 100000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model.to(DEVICE)
model.load_state_dict(torch.load("./recvae_no_aug_0.0001.pkl"))
for epoch in range(EPOCH):
    train.train(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        epoch=epoch + 1,
        gamma=1,
        dropout_rate=0.5,
    )

    print(train.eval(model=model, test_loader=eval_loader))

    torch.save(model.state_dict(), "./recvae_no_aug_0.000051.pkl")

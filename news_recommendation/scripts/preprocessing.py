import random
import numpy as np

import torch
import pandas as pd
from scipy.sparse import csr_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.optim as optim

from news_recommendation.utils import (
    train_test_split,
    dataset,
    early_stopping,
    augmentation,
)


BATCH_SIZE = 128


data = pd.read_csv("news_recommendation/data/view_log.csv")

# label encoding
user_encoder = dataset.load_label_encoder(
    pkl_path="news_recommendation/utils/user_encoder.pkl", data=data["userID"]
)
article_encoder = dataset.load_label_encoder(
    pkl_path="news_recommendation/utils/article_encoder.pkl", data=data["articleID"]
)

data["userID"] = user_encoder.transform(data["userID"])
data["articleID"] = article_encoder.transform(data["articleID"])

# train, test data split + get all article list
train_data, test_data, all_articles = (
    train_test_split.transfrom_and_split_test_and_train_data(data)
)
# data augmentation
# train_data = augmentation.augmentate_data(
#     data=train_data, target_label="articleID", all_items=all_articles
# )

# set dataset
test_data = train_test_split.split_y_from_data(data=test_data, label="articleID")

train_interaction_matrix = train_test_split.create_interaction_matrix(
    train_data, "articleID", all_articles
).toarray()
test_x_interaction_matrix = train_test_split.create_interaction_matrix(
    test_data, "articleID", all_articles
).toarray()
test_y_interaction_matrix = train_test_split.create_interaction_matrix(
    test_data, "y", all_articles
).toarray()


train_dataset = dataset.NewsDataset(train_interaction_matrix, None, mode="train")
test_dataset = dataset.NewsDataset(test_x_interaction_matrix, test_data.y, mode="valid")


# set dataloader
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

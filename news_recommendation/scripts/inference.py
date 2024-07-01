import random
import numpy as np
import copy


import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from news_recommendation.models import bivae
from news_recommendation.scripts import preprocessing

from news_recommendation.utils import dataset, train_test_split


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(42)
inference_loader = preprocessing.train_dataloader

INPUT_DIM = len(inference_loader.dataset[0])


model = bivae.VAE(hidden_dim=600, latent_dim=200, input_dim=INPUT_DIM)


device = "cuda" if torch.cuda.is_available() else "cpu"
model.load_state_dict(torch.load("./recvae_no_aug_0.000051.pkl"))

model.to(device)
model.eval()


predicted = []

with torch.no_grad():
    for inputs in inference_loader:
        predict = model(inputs, calculate_loss=False).cpu()

        for i in range(len(predict)):
            top_k_predictions = np.argsort(-predict[i].cpu().numpy())[:5]
            predicted_set = top_k_predictions.tolist()
            predicted.append(predicted_set)

print(len(predicted))

userID = []
articleID = []

for idx, item in enumerate(predicted):
    userID.append(idx)
    articleID.append(item)

data = pd.read_csv("news_recommendation/data/view_log.csv")

# label encoding
user_encoder = dataset.load_label_encoder(
    pkl_path="news_recommendation/utils/user_encoder.pkl", data=data["userID"]
)
article_encoder = dataset.load_label_encoder(
    pkl_path="news_recommendation/utils/article_encoder.pkl", data=data["articleID"]
)

submission_df = pd.DataFrame({"userID": userID, "articleID": articleID})
submission_df = submission_df.explode("articleID").reset_index(drop=True)
submission_df["userID"] = user_encoder.inverse_transform(submission_df["userID"])
submission_df["articleID"] = article_encoder.inverse_transform(
    submission_df["articleID"].astype(int)
)

submission_df.to_csv("submission_recvae_noaug_hidden600_latent200.csv")

import random
import numpy as np
import copy

import torch
import pandas as pd
from scipy.sparse import csr_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from news_recommendation.scripts import ease_train
from news_recommendation.models import bivae, ease
from news_recommendation.utils import (
    train_test_split,
    dataset,
    early_stopping,
    augmentation,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(42)

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

inference_data = copy.deepcopy(train_data)

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

inference_interaction_matrix = train_test_split.create_interaction_matrix(
    inference_data, "articleID", all_articles
).toarray()

user_similarity = cosine_similarity(train_interaction_matrix)
user_euclidean_similarity = euclidean_distances(train_interaction_matrix)


# 추천 점수 계산
user_predicted_scores = (
    user_similarity.dot(train_interaction_matrix)
    / np.array([np.abs(user_similarity).sum(axis=1)]).T
)

user_euclidean_scores = (
    user_euclidean_similarity.dot(train_interaction_matrix)
    / np.array([np.abs(user_euclidean_similarity).sum(axis=1)]).T
)


ease_output = ease_train.output.cpu().numpy()

mean_predicted_score = (user_predicted_scores + user_euclidean_scores + ease_output) / 3
predicted = []
user = []
for i in range(len(mean_predicted_score)):
    top_k_predictions = np.argsort(-mean_predicted_score[i])[:5]
    predicted.append(top_k_predictions.tolist())
    user.append(i)


submission = pd.DataFrame({"userID": user, "articleID": predicted})
submission = submission.explode("articleID").reset_index(drop=True)
submission["userID"] = user_encoder.inverse_transform(submission["userID"])
submission["articleID"] = article_encoder.inverse_transform(
    submission["articleID"].astype(int)
)

submission.to_csv("submission_ensemble_cf_euclidean_cosine_ease_noaug.csv")

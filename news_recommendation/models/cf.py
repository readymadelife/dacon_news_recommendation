import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

view_log_train = pd.read_csv("news_recommendation/data/view_log.csv")

# 사용자-기사 행렬 생성
user_article_matrix = (
    view_log_train.groupby(["userID", "articleID"]).size().unstack(fill_value=0)
)

# 사용자 간의 유사성 계산
user_similarity = cosine_similarity(user_article_matrix)

print(len(user_similarity))

# 추천 점수 계산
user_predicted_scores = (
    user_similarity.dot(user_article_matrix)
    / np.array([np.abs(user_similarity).sum(axis=1)]).T
)

# 이미 조회한 기사 포함해서 추천
recommendations = []
for idx, user in enumerate(user_article_matrix.index):
    # 해당 사용자의 추천 점수 (높은 점수부터 정렬)
    sorted_indices = user_predicted_scores[idx].argsort()[::-1]
    top5recommend = [
        article for article in user_article_matrix.columns[sorted_indices]
    ][:5]

    for article in top5recommend:
        recommendations.append([user, article])

# sample_submission.csv 형태로 DataFrame 생성
top_recommendations = pd.DataFrame(recommendations, columns=["userID", "articleID"])

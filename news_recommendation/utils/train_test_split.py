import random
import copy

import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix


def transfrom_and_split_test_and_train_data(data: DataFrame):
    user_item = data.groupby("userID")["articleID"].apply(list).reset_index()
    item_pool = data["articleID"].unique()
    user_item["isTest"] = user_item["articleID"].apply(
        lambda x: True if len(x) >= 6 and random.random() < 0.2 else False
    )
    test_data = (
        user_item[user_item["isTest"]].drop(columns=["isTest"]).reset_index(drop=True)
    )

    train_data = user_item.drop(columns=["isTest"]).reset_index(drop=True)

    return train_data, test_data, item_pool


def inverse_transform_data(data: DataFrame, label: str):
    data = data.explode(label).reset_index(drop=True)
    return data


def split_y_from_data(data: DataFrame, label):
    data = copy.deepcopy(data)
    data["y"] = data[label].str[-5:]
    data[label] = data[label].str[:-5]
    return data


def create_interaction_matrix(data: pd.DataFrame, item_col: str, all_items: np.ndarray):
    num_of_users = len(data)
    num_of_items = len(all_items)

    rows = data.index.repeat(data[item_col].str.len()).values
    cols = np.concatenate(data[item_col].values)
    interaction = np.ones(len(cols), dtype=int)

    interaction_matrix = csr_matrix(
        (interaction, (rows, cols)), shape=(num_of_users, num_of_items)
    )
    return interaction_matrix


def split_test_train_data(data: DataFrame, label: str):
    ids = data[label].unique()
    train_ids, test_ids = train_test_split(ids, test_size=0.2, random_state=42)
    train_data = data[data[label].isin(train_ids)]
    test_data = data[data[label].isin(test_ids)]
    return train_data, test_data

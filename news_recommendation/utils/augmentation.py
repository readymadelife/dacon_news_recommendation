import random
import copy
import math

import numpy as np
import pandas as pd


def add_subset(
    user_items: list,
    min_size: int,
):
    size = random.randint(min_size, len(user_items))
    start_pos = random.randint(0, len(user_items) - size)
    return user_items[start_pos : start_pos + size]


def add_noise(
    all_items: np.ndarray,
    user_items: list,
    noise_level: float = 0.2,
):
    user_item_copy = copy.deepcopy(user_items)
    num_noisy_item = math.ceil(len(user_item_copy) * noise_level)
    for _ in range(num_noisy_item):
        item = random.choice(all_items)
        while item in user_items:
            item = random.choice(all_items)

        item = int(item)
        user_item_copy.pop(0)
        insert_index = random.randint(0, len(user_item_copy))
        user_item_copy.insert(insert_index, item)
    return user_item_copy


def augmentate_data(data: pd.DataFrame, target_label: str, all_items: np.ndarray):
    data = copy.deepcopy(data)
    interactions = data[target_label].tolist()
    aug_data = []
    for item_list in interactions:
        if len(item_list) <= 2:
            continue

        elif len(item_list) < 5:
            for _ in range(3):
                subset = add_subset(item_list, 2)
                noisy = add_noise(all_items, item_list, 0.2)
                aug_data.append(subset)
                aug_data.append(noisy)
        else:
            for _ in range(15):
                subset = add_subset(item_list, 2)
                noisy = add_noise(all_items, item_list, 0.2)
                aug_data.append(subset)
                aug_data.append(noisy)
    augmented_data = interactions + aug_data
    index = [i + len(interactions) for i in range(len(aug_data))]
    augmented_id = data["userID"].tolist() + index
    new_df = pd.DataFrame({"userID": augmented_id, target_label: augmented_data})
    return new_df

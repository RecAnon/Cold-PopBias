import numpy as np
from util.databuilder import ColdStartDataBuilder
from util.loader import DataLoader
import os
import pickle
import torch

def get_data(dataset):
    feat_filenames = sorted(os.listdir(f"./data/{dataset}/feats"))
    feat_files = [f"./data/{dataset}/feats/{f}" for f in feat_filenames]
    item_content = torch.from_numpy(
        [np.load(f).astype(np.float32) for f in feat_files][0]
    )
    training_data = DataLoader.load_data_set(
        f"./data/{dataset}/cold_item/warm_train.csv"
    )
    # following the widely used setting in previous works, the 'all' set is used for validation.
    all_valid_data = DataLoader.load_data_set(
        f"./data/{dataset}/cold_item/overall_val.csv"
    )
    warm_valid_data = DataLoader.load_data_set(
        f"./data/{dataset}/cold_item/warm_val.csv"
    )
    cold_valid_data = DataLoader.load_data_set(
        f"./data/{dataset}/cold_item/cold_item_val.csv"
    )
    all_test_data = DataLoader.load_data_set(
        f"./data/{dataset}/cold_item/overall_test.csv"
    )
    warm_test_data = DataLoader.load_data_set(
        f"./data/{dataset}/cold_item/warm_test.csv"
    )
    cold_test_data = DataLoader.load_data_set(
        f"./data/{dataset}/cold_item/cold_item_test.csv"
    )

    data_info_dict = pickle.load(
        open(f"./data/{dataset}/cold_item/info_dict.pkl", "rb")
    )
    user_num = data_info_dict["user_num"]
    item_num = data_info_dict["item_num"]
    warm_user_idx = data_info_dict["warm_user"]
    warm_item_idx = data_info_dict["warm_item"]
    cold_user_idx = data_info_dict["cold_user"]
    cold_item_idx = data_info_dict["cold_item"]

    data = ColdStartDataBuilder(
        training_data,
        warm_valid_data,
        cold_valid_data,
        all_valid_data,
        warm_test_data,
        cold_test_data,
        all_test_data,
        user_num,
        item_num,
        warm_user_idx,
        warm_item_idx,
        cold_user_idx,
        cold_item_idx,
        None,
        item_content,
    )
    return data, item_content

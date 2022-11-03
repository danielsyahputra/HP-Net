import os
import random
import pickle
import argparse
import numpy as np
from tqdm.auto import tqdm
from scipy.io import loadmat

np.random.seed(0)
random.seed(0)

def generate_data_description(save_dir: str) -> None:
    dataset = {}
    dataset['description'] = "pa100k"
    dataset['root'] = "./data/pa100k"
    dataset["image"] = []
    dataset["att"] = []
    dataset["att_name"] = []
    dataset["selected_attribute"] = range(26)

    data = loadmat(file_name="data/annotation/annotation.mat")
    for idx in tqdm(range(26)):
        dataset["att_name"].append(data['attributes'][idx][0][0])

    for idx in tqdm(range(80000)):
        dataset["image"].append(data['train_images_name'][idx][0][0])
        dataset["att"].append(data['train_label'][idx, :].tolist())

    for idx in tqdm(range(10000)):
        dataset["image"].append(data['val_images_name'][idx][0][0])
        dataset["att"].append(data['val_label'][idx, :].tolist())

    for idx in tqdm(range(10000)):
        dataset['image'].append(data['test_images_name'][idx][0][0])
        dataset['att'].append(data['test_label'][idx, :].tolist())

    with open(os.path.join(save_dir, 'pa100k.pkl'), 'wb') as f:
        pickle.dump(dataset, f)

def split_dataset(partition_path: str) -> None:
    partition = {}
    partition["train_val"] = []
    partition["train"] = []
    partition["val"] = []
    partition["test"] = []
    partition["weight_train_val"] = []
    partition["weight_train"] = []

    data = loadmat(file_name="data/annotation/annotation.mat")
    train = list(range(80000))
    val = [i + 80000 for i in range(10000)]
    test = [i + 90000 for i in range(10000)]
    train_val = train + val
    partition["train"].append(train)
    partition["val"].append(val)
    partition["train_val"].append(train_val)
    partition["test"].append(test)

    train_label = data["train_label"].astype("float32")
    train_val_label = np.concatenate((data['train_label'], data['val_label']), axis=0).astype('float32')
    weight_train = np.mean(train_label == 1, axis=0).tolist()
    weight_train_val = np.mean(train_val_label == 1, axis=0).tolist()

    partition["weight_train_val"].append(weight_train_val)
    partition["weight_train"].append(weight_train)

    with open(partition_path, "wb") as f:
        pickle.dump(partition, f)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="PA-100K Dataset")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="labels"
    )
    parser.add_argument(
        "--partition-path",
        type=str,
        default="labels/pa100k_partition.pkl"
    )
    args = parser.parse_args()
    save_dir = args.save_dir
    partition_path = args.partition_path
    generate_data_description(save_dir=save_dir)
    split_dataset(partition_path=partition_path)
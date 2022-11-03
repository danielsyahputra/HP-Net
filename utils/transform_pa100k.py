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

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="PA-100K Dataset")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="labels"
    )
    args = parser.parse_args()
    save_dir = args.save_dir
    generate_data_description(save_dir=save_dir)
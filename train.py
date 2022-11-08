import torch
import pickle
import argparse
import torch.nn as nn
from data.dataset import dataloader
from utils.engine import train_model

from typing import Iterable

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-model",
        default=None,
        choices=["MainNet", "AF1", "AF2", "AF3", "HP"],
        help="Choose model"
    )

    parser.add_argument(
        "-num-workers",
        default=2,
        type=int,
        help="Number of workers"
    )
    parser.add_argument(
        "-batch-size",
        default=1,
        type=int
    )
    parser.add_argument(
        "-lr",
        default=0.001,
        type=float
    )
    args = parser.parse_args()
    return args

def get_loss_fn():
    with open("assets/loss_cls_weight.pkl", "rb") as f:
        loss_cls_weight = pickle.load(f)
    weight = torch.Tensor(loss_cls_weight)
    loss_fn = nn.BCEWithLogitsLoss(weight=weight)
    return loss_fn

def  main(args):
    model_name = args.model
    batch_size = args.batch_size
    num_workers = args.num_workers
    learning_rate = args.lr
    train_loader, val_loader, test_loader = dataloader(
        dataset_path="labels/pa100k.pkl",
        partition_path="labels/pa100k_partition.pkl",
        batch_size=batch_size,
        num_workers=num_workers
    )
    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    train_model(
        model_name=model_name,
        loaders=loaders,
        loss_fn=get_loss_fn(),
        epochs=10,
        mnet_path="checkpoint/MainNet",
        afnet_path=[None, None, None],
        lr=learning_rate 
    )
    

if __name__=="__main__":
    args = parse_args()
    main(args=args)
import torch
import pickle
import argparse
import torch.nn as nn
from data.dataset import dataloader
from utils.engine import train_model

def parse_args():
    parser = argparse.ArgumentParser()

    # Choose model
    parser.add_argument(
        "-model",
        default=None,
        choices=["MainNet", "AF1", "AF2", "AF3", "HP"],
        help="Choose model"
    )

    # Pre-trained checkpoint
    parser.add_argument("-r", default=False, help="Resume training or not")
    parser.add_argument("-checkpoint", default=None, help="Load weight path")
    parser.add_argument("-mnet-path", default=None, help="MainNet weight path")
    parser.add_argument("-af1-path", default=None, help="AF1 weight path")
    parser.add_argument("-af2-path", help="AF2 weight path", default=None)
    parser.add_argument("-af3-path", default=None, help="AF3 weight path")

    parser.add_argument(
        "-num-workers",
        default=2,
        type=int,
        help="Number of workers"
    )
    parser.add_argument(
        "-batch-size",
        default=128,
        type=int
    )
    parser.add_argument(
        "-lr",
        default=0.001,
        type=float
    )
    parser.add_argument(
        "-mGPUs",
        help="Whether to use multiple GPU",
        action="store_true"
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
    checkpoint=args.checkpoint
    mnet_path = args.mnet_path
    af1_path = args.af1_path
    af2_path = args.af2_path
    af3_path = args.af3_path
    batch_size = args.batch_size
    num_workers = args.num_workers
    learning_rate = args.lr
    resume = args.r
    mGPUs = args.mGPUs

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
        epochs=5,
        mnet_path=mnet_path,
        afnet_path=[af1_path, af2_path, af3_path],
        lr=learning_rate,
        resume=resume,
        mGPUs=mGPUs,
        checkpoint=checkpoint
    )
    

if __name__=="__main__":
    args = parse_args()
    main(args=args)
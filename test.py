import argparse
from data.dataset import dataloader
from utils.engine import test_model

def parse_args():
    parser = argparse.ArgumentParser(description="Args for Testing HP-Net")
    parser.add_argument("-model", choices=["MainNet", "AF1", "AF2", "AF3", "HP"], help="Choose model")
    parser.add_argument("-p", help="Weight file path", required=True)
    parser.add_argument(
        "-att",
        help="Attention Results",
        choices=[
            "no_att",
            "img_save",
            "img_show",
            "pkl_save"
        ],
        default="no_att"
    )
    args = parser.parse_args()
    return args



def main(args):
    test_loader = dataloader(
        dataset_path="labels/pa100k.pkl",
        partition_path="labels/pa100k_partition.pkl",
        split="test",
        batch_size=1,
        num_workers=2
    )
    weight_path = args.p
    att_mode = args.att
    model_name = args.model
    test_model(
        model_name=model_name,
        loader=test_loader,
        att_mode=att_mode,
        weight_path=weight_path
    )

if __name__=="__main__":
    args = parse_args()
    main(args=args)
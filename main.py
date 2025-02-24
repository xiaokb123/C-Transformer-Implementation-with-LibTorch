# main.py
import argparse
from train.train import train
from test.test import test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer Training and Testing")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Mode: train or test")
    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "test":
        test()
    else:
        print("Invalid mode. Use --mode train or --mode test.")
# ===================================================
# File: split_dataset.py
# Author: David Kedra
# ===================================================

import argparse
import random
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to raw data file (tab-separated src\tgt).")
    parser.add_argument("--n_train", type=int, default=-1)
    parser.add_argument("--n_test", type=int, default=0)
    parser.add_argument("--unshuffled", action="store_true")
    parser.add_argument("--random_seed", type=int, default=1717)
    parser.add_argument("--range_train", type=str, default=None)
    parser.add_argument("--range_test", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="data")
    args = parser.parse_args()

    random.seed(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load raw data
    with open(args.data_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Optional shuffling
    if not args.unshuffled:
        random.shuffle(lines)

    # Select train/test ranges
    if args.range_train:
        start, end = map(int, args.range_train.split(":"))
        train_lines = lines[start:end]
    else:
        train_lines = lines[:args.n_train]

    if args.range_test:
        start, end = map(int, args.range_test.split(":"))
        test_lines = lines[start:end]
    else:
        test_lines = lines[args.n_train:args.n_train + args.n_test]

    # Save
    with open(os.path.join(args.output_dir, "train.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines))
    with open(os.path.join(args.output_dir, "test.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(test_lines))

    print(f"Saved {len(train_lines)} train and {len(test_lines)} test examples to {args.output_dir}")


if __name__ == "__main__":
    main()

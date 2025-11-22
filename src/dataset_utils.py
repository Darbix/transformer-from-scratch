# ===================================================
# File: dataset_utils.py
# Author: David Kedra
# ===================================================

import re


def parse_dataset(dataset_path, swap=False):
    """
    Parse a dataset of translated sentences into a list of (SRC, TGT) sentence pairs.

    Each line in the dataset should be TAB-separated with at least two columns:
    target sentence (e.g., EN), source sentence (e.g., CZ), etc.

    Args:
        dataset_path (str): Path to the dataset file.

    Returns:
        list of tuples: Each tuple is a pair of SRC sentence & TGT sentence or
                        a pair of TGT sentence & SRC sentence if swap=True.
    """
    # Ensure the file exists
    try:
        f = open(dataset_path, 'r', encoding='utf-8')
        f.close()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {dataset_path}")

    data_pairs = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Split sentences by tabs and remove leading and trailing spaces
            parts = [p.strip() for p in re.split(r'\t+', line.strip())]

            # Make sure there are at least 2 columns (src, tgt)
            if len(parts) >= 2:
                tgt = parts[0].strip()
                src = parts[1].strip()

                if(swap):
                    data_pairs.append((tgt, src))
                else:
                    data_pairs.append((src, tgt))

    if not data_pairs:
        raise ValueError("Cannot parse any data in a form <tgt_sentence>\\t<src_sentence>")

    return data_pairs

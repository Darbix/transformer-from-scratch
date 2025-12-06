# ===================================================
# File: test.py
# Author: David Kedra
# ===================================================

import argparse
import torch
import sys
import os

from src.transformer import Transformer
from src.tokenizer import TextTokenizer
from src.dataset_utils import parse_dataset
from src.utils import load_model, bleu_n, fix_token_spacing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/test.txt")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--max_tgt_len", type=int, default=-1)
    parser.add_argument("--swap", action="store_true",
                        help="Loads data as <SRC>\\t<TGT> instead of <TGT>\\t<SRC>.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----- Load model + tokenizer state -----
    model, src_tokenizer, tgt_tokenizer, config = load_model(
        args.checkpoint_dir,
        device=device
    )

    # ----- Load dataset -----
    data_pairs = parse_dataset(args.data_path, swap=args.swap)
    czech_sentences, english_sentences = map(list, zip(*data_pairs))

    # If max_tgt_len is not specified, use the max length of target sequences
    max_tgt_len = args.max_tgt_len
    if(max_tgt_len <= 0):
        max_tgt_len = tgt_tokenizer.get_max_seq_len(english_sentences)

    sos_id = tgt_tokenizer.vocab["<SOS>"]
    eos_id = tgt_tokenizer.vocab["<EOS>"]

    # BLEU accumulators
    bleu1_scores = []
    bleu2_scores = []
    bleu3_scores = []

    model.eval()

    # ----- Iterate through all samples -----
    for i, (src, tgt) in enumerate(data_pairs):
        # Encode the source input
        src_ids = torch.tensor([src_tokenizer.encode(src)], device=device)

        # Predict the target sentence
        with torch.no_grad():
            pred_ids = model.inference(
                src_ids,
                sos_id,
                eos_id,
                max_tgt_len=max_tgt_len
            )

        # Convert predicted IDs back to tokens
        pred_tokens = tgt_tokenizer.seq_ids2tokens(pred_ids)

        # ----- BLEU metric computation -----
        pred_text = fix_token_spacing(" ".join(pred_tokens))
        ref_text = tgt
        
        bleu1 = bleu_n(pred_text, ref_text, max_n=1)
        bleu2 = bleu_n(pred_text, ref_text, max_n=2)
        bleu3 = bleu_n(pred_text, ref_text, max_n=3)

        bleu1_scores.append(bleu1)
        bleu2_scores.append(bleu2)
        bleu3_scores.append(bleu3)

        print(f"[{i}]", "-" * (46-(i//10)))
        print(f"SRC:  {src}")
        print(f"PRED: {pred_text}")
        print(f"REF:  {ref_text}")
        print(f"BLEU-1: {bleu1:.3f}, BLEU-2: {bleu2:.3f}, BLEU-3: {bleu3:.3f}")

    # ----- Overall BLEU scores -----
    avg_bleu1 = sum(bleu1_scores) / len(bleu1_scores)
    avg_bleu2 = sum(bleu2_scores) / len(bleu2_scores)
    avg_bleu3 = sum(bleu3_scores) / len(bleu3_scores)

    print("=" * 50)
    print(f"FINAL BLEU-1: {avg_bleu1:.4f}")
    print(f"FINAL BLEU-2: {avg_bleu2:.4f}")
    print(f"FINAL BLEU-3: {avg_bleu3:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()

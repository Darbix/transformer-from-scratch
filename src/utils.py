# ===================================================
# File: utils.py
# Author: David Kedra
# ===================================================

import torch
import json
import math
import os
import re

from .tokenizer import TextTokenizer
from .transformer import Transformer


def save_model(model, src_tokenizer, tgt_tokenizer, config, save_dir):
    """
    Save model weights (checkpoint) & configuration and both tokenizers to files.

    Args:
        model: Trained transformer model.
        src_tokenizer: Source sentence tokenizer.
        tgt_tokenizer: Target sentence tokenizer.
        config: Model configuration dictionary.
        save_dir: Directory to save the model files.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

    # Save model configuration
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Save tokenizers
    src_tokenizer.save(os.path.join(save_dir, "src_tokenizer.json"))
    tgt_tokenizer.save(os.path.join(save_dir, "tgt_tokenizer.json"))

    print(f"Model, tokenizers and configuration saved to {save_dir}")


def load_model(save_dir, device="cpu"):
    """
    Load model, tokenizers, and config from a save directory.

    Args:
        save_dir: Directory where the checkpoint files are saved.
        device: Device to load the model on.
    Returns:
        model, src_tokenizer, tgt_tokenizer, config
    """
    # Load tokenizers
    src_tokenizer = TextTokenizer.load(os.path.join(save_dir,
                                                    "src_tokenizer.json"))
    tgt_tokenizer = TextTokenizer.load(os.path.join(save_dir,
                                                    "tgt_tokenizer.json"))
    
    # Load model configuration
    with open(os.path.join(save_dir, "config.json"), "r") as f:
        config = json.load(f)

    # Re-create model using config
    model = Transformer(
        config["src_vocab_size"],
        config["tgt_vocab_size"],
        config["d_model"],
        config["num_heads"],
        config["num_encoder_layers"],
        config["num_decoder_layers"],
        config["d_ff"],
        config["dropout"],
        config["max_seq_length"],
        use_rope=config["use_rope"]
    ).to(device)

    # Load weights
    state_dict = torch.load(os.path.join(save_dir, "model.pt"),
                            map_location=device)
    model.load_state_dict(state_dict)

    model.eval()

    return model, src_tokenizer, tgt_tokenizer, config


def bleu_n(pred_tokens, ref_tokens, max_n):
    """
    N-gram precision metric (degree of similarity).
    Compute cumulative BLEU-N score for a single predicted sentence
    against a reference. BLEU is a proportion of correct predicted n-grams
    to the reference n-grams.

    Args:
        pred_tokens: list of predicted tokens
        ref_tokens: list of reference tokens
        max_n: max n-gram order (1=unigram, 2=bigram, etc.)

    Returns:
        BLEU-n score (0.0 - 1.0)
    """
    def _ngrams(tokens, n):
        """Return a list of n-grams from a list of tokens."""
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def _count_ngrams(ngrams_list):
        """Aggregate occurrences of n-grams to a dictionary"""
        counts = {}
        for ngram in ngrams_list:
            if ngram in counts:
                counts[ngram] += 1
            else:
                counts[ngram] = 1
        return counts

    def normalize(tokens):
        """Removes spaces before punctuation"""
        text = " ".join(tokens)
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        return text.split()
    
    pred_tokens = normalize(pred_tokens)
    ref_tokens = normalize(ref_tokens)

    precisions = []

    # Step 1: Calculate a precision per each n size of n-gram
    for n in range(1, max_n + 1):
        # Extract n-grams to a list
        pred_ngrams = _ngrams(pred_tokens, n)
        ref_ngrams = _ngrams(ref_tokens, n)

        # Create a dictionary of unique n-gram counts
        pred_counts = _count_ngrams(pred_ngrams)
        ref_counts = _count_ngrams(ref_ngrams)

        # Clip counts: calculate the number of overlapping n-gram occurrences
        overlap = 0
        for ngram, count in pred_counts.items():
            if ngram in ref_counts:
                overlap += min(count, ref_counts[ngram])
        # Count the total number of n-gram positions
        total = max(1, sum(pred_counts.values()))

        # Avoid log(0)
        precision = overlap / total
        precision = precision if precision > 0 else 1e-9

        # Append the score
        precisions.append(precision)

    # Step 4: Geometric mean of precisions (weighted sum)
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)

    c = max(len(pred_tokens), 1)
    r = len(ref_tokens)
    bp = 1.0 if c > r else math.exp(1 - r / c)

    # Step 4: Full BLEU score with a geometric mean of particular scores
    bleu = bp * geo_mean

    return bleu

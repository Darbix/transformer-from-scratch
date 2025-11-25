# ===================================================
# File: train.py
# Author: David Kedra
# ===================================================

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys

from src.tokenizer import TextTokenizer
from src.transformer import Transformer
from src.dataset_utils import parse_dataset
from src.utils import save_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/train.txt")
    parser.add_argument("--swap", action="store_true",
                        help="Loads data as <SRC>\\t<TGT> instead of <TGT>\\t<SRC>.")
    parser.add_argument("--tokenizer", type=str, default="BPE", choices=["WORD","BPE"])
    parser.add_argument("--max_vocab_size", type=int, default=-1)
    parser.add_argument("--d_model", type=int, default=48)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--runs_dir", type=str, default="runs")
    args = parser.parse_args()

    # Device CPU/GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load training data
    data_pairs = parse_dataset(args.data_path, swap=args.swap)
    src_sentences, tgt_sentences = map(list, zip(*data_pairs))

    # Tokenizers
    src_tokenizer = TextTokenizer(src_sentences, mode=args.tokenizer,
                                  max_vocab_size=args.max_vocab_size)
    tgt_tokenizer = TextTokenizer(tgt_sentences, mode=args.tokenizer,
                                  max_vocab_size=args.max_vocab_size)

    src_vocab_size = src_tokenizer.vocab_size()
    tgt_vocab_size = tgt_tokenizer.vocab_size()
    max_src_len = src_tokenizer.get_max_seq_len(src_sentences)
    max_tgt_len = tgt_tokenizer.get_max_seq_len(tgt_sentences)

    print(f"SRC vocabulary size: {src_vocab_size}, max. sequence length: {max_src_len}")
    print(f"TGT vocabulary size: {tgt_vocab_size}, max. sequence length: {max_tgt_len}")

    # Encode data
    src_data = torch.tensor(
        [src_tokenizer.encode(s, max_src_len) for s in src_sentences], device=device)
    tgt_data = torch.tensor(
        [tgt_tokenizer.encode(s, max_tgt_len) for s in tgt_sentences], device=device)

    print(f"SRC data shape: {src_data.shape}")
    print(f"TGT data shape: {tgt_data.shape}")

    if(args.shuffle):
        # Shuffle data using vectorized index permutation
        perm = torch.randperm(len(src_data))
        src_data = src_data[perm]
        tgt_data = tgt_data[perm]

    # Model
    max_seq_length = max(max_src_len, max_tgt_len)
    model = Transformer(
        src_vocab_size, tgt_vocab_size, args.d_model, args.num_heads,
        args.num_layers, args.num_layers, args.d_ff, args.dropout, max_seq_length
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.98), eps=1e-9)
    model.train()

    loss_history = []

    batch_size = args.batch_size
    num_batches = (len(src_data) + batch_size - 1) // batch_size

    for epoch in range(args.epochs):

        epoch_loss = 0
        for i in range(num_batches):
            # Get indices of sentences in the batch
            start = i * batch_size
            end = min((i + 1) * batch_size, len(src_data))

            encoder_input = src_data[start:end]
            decoder_input = tgt_data[start:end, :-1] # Shifted as (<SOS>, ...)
            true_labels = tgt_data[start:end, 1:]    # Shifted as (..., <EOS>)

            optimizer.zero_grad()
            # encoder_input shape: (batch_size, src_seq_len)
            # decoder_input shape: (batch_size, tgt_seq_len)
            # output shape:        (batch_size, tgt_seq_len, tgt_vocab_size)
            output = model(encoder_input, decoder_input)
            
            # Reshape to (batch_size * tgt_seq_len, tgt_vocab_size)
            predictions = output.contiguous().view(-1, tgt_vocab_size)
            # Reshape to (batch_size * tgt_seq_len)
            true_labels = true_labels.contiguous().view(-1)

            loss = criterion(predictions, true_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / num_batches
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")




    # Save model
    os.makedirs(args.runs_dir, exist_ok=True)
    save_dir = os.path.join(args.runs_dir, f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(save_dir, exist_ok=True)

    config = {
        "src_vocab_size": src_vocab_size,
        "tgt_vocab_size": tgt_vocab_size,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "num_encoder_layers": args.num_layers,
        "num_decoder_layers": args.num_layers,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
        "max_seq_length": max_seq_length,
        "use_rope": True
    }

    save_model(model, src_tokenizer, tgt_tokenizer, config, save_dir)

    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Transformer Training Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir,"training_loss.png"))
    print(f"Training done. Model and plot saved to {save_dir}")

if __name__ == "__main__":
    main()

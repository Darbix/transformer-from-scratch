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
    parser.add_argument("--max_seq_len", type=int, default=-1,
                    help="Truncate sequences of tokens to this length.")
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
    parser.add_argument("--print_model", action="store_true")
    parser.add_argument("--use_rope", action="store_true")
    parser.add_argument("--val_path", type=str, default=None,
                    help="Optional validation dataset.")
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

    if(args.max_seq_len > 0):
        max_src_len = args.max_seq_len
        max_tgt_len = args.max_seq_len
        print(f"Max sequence length is set to {args.max_seq_len}")

    # Encode training data
    src_data = torch.tensor(
        [src_tokenizer.encode(s, max_src_len) for s in src_sentences],
        device=device
    )
    tgt_data = torch.tensor(
        [tgt_tokenizer.encode(s, max_tgt_len) for s in tgt_sentences],
        device=device
    )

    print(f"SRC data shape: {src_data.shape}")
    print(f"TGT data shape: {tgt_data.shape}")

    if(args.shuffle):
        # Shuffle data using vectorized index permutation
        perm = torch.randperm(len(src_data))
        src_data = src_data[perm]
        tgt_data = tgt_data[perm]

    # Load and encode validation data
    val_src_data = val_tgt_data = None
    if args.val_path is not None:
        val_pairs = parse_dataset(args.val_path, swap=args.swap)
        val_src_sentences, val_tgt_sentences = map(list, zip(*val_pairs))
        
        # Tokenize sentences
        val_src_data = torch.tensor(
            [src_tokenizer.encode(s, max_src_len) for s in val_src_sentences],
            device=device
        )
        val_tgt_data = torch.tensor(
            [tgt_tokenizer.encode(s, max_tgt_len) for s in val_tgt_sentences],
            device=device
        )

        print(f"Loaded validation set: {len(val_src_data)} samples from {args.val_path}")


    # Model
    max_seq_length = max(max_src_len, max_tgt_len)
    model = Transformer(
        src_vocab_size, tgt_vocab_size, args.d_model, args.num_heads,
        args.num_layers, args.num_layers, args.d_ff, args.dropout,
        max_seq_length, use_rope=args.use_rope
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.98), eps=1e-9)
    model.train()

    loss_history = []
    val_loss_history = []

    batch_size = args.batch_size
    num_batches = (len(src_data) + batch_size - 1) // batch_size

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,} ({total_params/1e6:.2f}M)")
 
    if(args.print_model):
        print("")
        print(model)
    
    # Configuration dictionary used for model checkpoints
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
        "use_rope": args.use_rope
    }

    # Create directory for model checkpoints
    os.makedirs(args.runs_dir, exist_ok=True)

    print("")
    print("Start of training")

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

        avg_train_loss = epoch_loss / num_batches
        loss_history.append(avg_train_loss)

        # Validation
        if val_src_data is not None:
            val_loss = compute_val_loss(
                model, val_src_data, val_tgt_data, batch_size,
                criterion, tgt_vocab_size
            )
            val_loss_history.append(val_loss)
            print(f"Epoch {epoch+1} | Train loss: {avg_train_loss:.4f} | Val loss: {val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1} | Train loss: {avg_train_loss:.4f}")
        
        # Save model checkpoint
        if((epoch+1) % 10 == 0 or (epoch+1) >= args.epochs):
            save_dir = os.path.join(args.runs_dir, f"checkpoint_E{epoch+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(save_dir, exist_ok=True)
            save_model(model, src_tokenizer, tgt_tokenizer, config, save_dir)
            print(f"Model, tokenizer and configuration were saved to {save_dir}")
    
    # Save model
    save_dir = os.path.join(args.runs_dir, f"checkpoint_last_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(save_dir, exist_ok=True)

    save_model(model, src_tokenizer, tgt_tokenizer, config, save_dir)

    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(loss_history, label="Training Loss")
    if(len(val_loss_history) > 0):
        plt.plot(val_loss_history, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Transformer Training & Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir,"loss.png"))

    print(f"Training done. The loss plot was saved to {save_dir}")


@torch.no_grad()
def compute_val_loss(model, val_src, val_tgt, batch_size, criterion, tgt_vocab_size):
    """Compute validation loss for the whole validation dataset."""
    model.eval()
    num_batches = (len(val_src) + batch_size - 1) // batch_size
    total_loss = 0

    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(val_src))

        enc_input = val_src[start:end]
        dec_input  = val_tgt[start:end, :-1]
        labels  = val_tgt[start:end, 1:]

        out = model(enc_input, dec_input)
        pred = out.contiguous().view(-1, tgt_vocab_size)
        labels = labels.contiguous().view(-1)

        loss = criterion(pred, labels)
        total_loss += loss.item()

    model.train()
    return total_loss / num_batches


if __name__ == "__main__":
    main()

# ====================================================================
# File: tokenizer.py
# Author: David Kedra
# ====================================================================

from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import torch
import json
import os


class TextTokenizer:

    MODE_WORD = "WORD"
    MODE_BPE = "BPE"

    def __init__(self, corpus, mode=MODE_WORD, max_vocab_size=-1):
        """
        Args:
            mode (str): 'WORD' (simple word-level tokenenizer) or 'BPE' (subword BPE tokenizer).
            corpus: list of raw string sentences.
            max_vocab_size (int): upper limit for the BPE vocabulary size.
        """
        self.mode = mode
        self.tokenizer = None   # Tokenizer model
        self.vocab = {}         # Dictionary of (token: ID) pairs
        self.dict_id2token = {} # Inverse dictionary (ID to token mapping)

        if not corpus:
            return

        # ----- WORD-LEVEL TOKENIZER -----
        if mode == self.MODE_WORD:
            # Create a vocabulary dictionary of pairs (word_token: ID)
            self.vocab = self._build_word_vocab(corpus)

        # ----- BPE TOKENIZER -----
        elif mode == self.MODE_BPE:
            self.tokenizer = Tokenizer(models.BPE())
            # Pre-tokenizer splits input text on whitespace to whole words
            self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

            trainer = trainers.BpeTrainer(
                vocab_size=max_vocab_size if max_vocab_size > 0 else 30000,
                special_tokens=["<PAD>", "<SOS>", "<EOS>"]
            )

            self.tokenizer.train_from_iterator(corpus, trainer=trainer)

            # Create a vocabulary dictionary of pairs (subword_token: ID)
            vocab_dict = self.tokenizer.get_vocab(with_added_tokens=True)
            # Sort by ID
            vocab_items = sorted(vocab_dict.items(), key=lambda x: x[1])
            self.vocab = {tok: idx for tok, idx in vocab_items}

        else:
            raise ValueError(f"the mode must be '{self.MODE_WORD}' or '{self.MODE_BPE}'")

        # Create an inversed vocabulary with pairs (ID: token)
        self.dict_id2token = {id: token for token, id in self.vocab.items()}

    def _build_word_vocab(self, corpus):
        """
        Build a vocabulary of whole words and special tokens.

        Args:
            corpus: A list of text sentences.
        """
        vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
        idx = 3
        for sentence in corpus:
            for word in sentence.split():
                if word not in vocab:
                    vocab[word] = idx
                    idx += 1
        return vocab

    def encode(self, sentence, pad_to_len=None):
        """
        Encode a sentence into a sequence of token IDs

        Args:
            sentence (str or list): Sentence to encode.
            pad_to_len (int): Desired final length of the sequence with padding.

        Returns:
            list [int]: List of encoded token IDs.
        """
        # ----- WORD-LEVEL TOKENIZATION -----
        if self.mode == self.MODE_WORD:
            # Split the sentence to words by whitespaces
            sentence = sentence.split()
            ids = [self.vocab["<SOS>"]] + [self.vocab[w] for w in sentence] + [self.vocab["<EOS>"]]
            if pad_to_len and len(ids) < pad_to_len:
                ids += [self.vocab["<PAD>"]] * (pad_to_len - len(ids))
            return ids

        # ----- BPE TOKENIZATION -----
        elif self.mode == self.MODE_BPE:
            sos, eos, pad = [self.tokenizer.token_to_id(t) for t in ("<SOS>", "<EOS>", "<PAD>")]
            enc = self.tokenizer.encode(sentence)
            # enc = self.tokenizer.encode(sentence)
            ids = [sos] + enc.ids + [eos]
            tokens = ["<SOS>"] + enc.tokens + ["<EOS>"]
            if pad_to_len and len(ids) < pad_to_len:
                ids += [pad] * (pad_to_len - len(ids))
            return ids

    def vocab_size(self):
        return len(self.vocab)

    def seq_ids2tokens(self, encoded_seq):
        """
        Converts the encoded sequence of token IDs back to token text words.

        Args:
            encoded_seq (list): List of token IDs.
        Returns:
            list [str]: List of token text words.
        """
        seq_tokens = [self.dict_id2token[id] for id in encoded_seq]
        return seq_tokens

    def get_max_seq_len(self, sentences):
        """
        Compute the maximum sequence length after tokenization (max. number of tokens).

        Args:
            sentences [str]: List of sentences.
        Returns:
            int: Maximum sequence length (number of tokens - words, subwords, etc.)
        """
        lengths = []

        for s in sentences:
            # Get the non-padded sequence of token IDs (includes <SOS> and <EOS>)
            ids = self.encode(s)
            lengths.append(len(ids))

        return max(lengths)

    def save(self, path):
        """
        Save tokenizer vocabulary to a JSON file. For BPE mode, also saves
        the tokenizer model metadata.

        Args:
            path (str): Path to save the tokenizer.
        """
        base = os.path.splitext(path)[0]
        vocab_path = base + "_vocab.json"
        meta_path = base + "_meta.json" # Only BPE

        # Save the mode and the vocabulary
        data = {"mode": self.mode, "vocab": self.vocab}
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # For BPE, also save the tokenizer model
        if self.mode == self.MODE_BPE and self.tokenizer is not None:
            self.tokenizer.save(meta_path)
    
    @staticmethod
    def load(path):
        """
        Load a tokenizer. It also restores the mode tokenizer for BPE mode.

        Args:
            path (str): Path to the tokenizer.
        Returns:
            TextTokenizer: Loaded tokenizer model.
        """

        base = os.path.splitext(path)[0]
        vocab_path = base + "_vocab.json"
        meta_path = base + "_meta.json" # Only BPE

        with open(vocab_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tok = TextTokenizer(corpus=None, mode=data["mode"])
        tok.vocab = data["vocab"]
        # Recreate the inverse dictionary
        tok.dict_id2token = {id: token for token, id in tok.vocab.items()}

        # Restore the tokenizer model for BPE
        if tok.mode == TextTokenizer.MODE_BPE:
            tok.tokenizer = Tokenizer.from_file(meta_path)

        return tok

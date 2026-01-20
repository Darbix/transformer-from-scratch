# Transformer from Scratch: Sequence-to-Sequence Model

<img src="imgs/transformer_architecture.png?" alt="Transformer" width="100%"/>

*Architecture diagram inspired by the original Transformer model from [Vaswani et al., “Attention Is All You Need” (2017)](https://arxiv.org/abs/1706.03762).*

## Table of Contents
- [Project Overview](#project-overview)
- [Technical Report](#technical-report)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [References](#references)

---

## Project Overview

This project implements a **Transformer-based sequence-to-sequence model** from scratch in PyTorch [4], closely following the architecture introduced in the paper “Attention Is All You Need” by Vaswani et al. (2017) [1]. The goal of the project is to provide a clear implementation that exposes the core building blocks of modern Transformer models without relying on high-level abstractions. The project demonstrates core techniques in modern **Natural Language Processing (NLP)**, including:

- Multi-head attention and cross-attention mechanisms
- Encoder-decoder architecture
- Word-level and subword-level (BPE) tokenization [5]
- Rotary Positional Embeddings (RoPE) [2]
- Autoregressive sequence generation for inference

The implementation was inspired by the structure presented in the article by Ebad Sayed [3]. The model can be trained on **any sequence-to-sequence dataset**, such as language translation, summarization, or other NLP tasks. In examples, it was trained on Czech-to-English sentence pairs from OPUS corpora (https://opus.nlpl.eu/), but the architecture is fully general-purpose (task-agnostic).

The model is trained using *cross-entropy* loss with padded sequences and evaluated using the **BLEU-N** metric.


**Example:**

| 📝 Input (Czech) | 🤖 Model Prediction | ✅ Reference Output |
|:--|:--|:--|
| Co se tam děje? | What's going on here? | What is happening there? |
| Tom je jediný, kdo může Mary pomoct. | Tom is the only person who can help Mary. | Tom is the only one who can help Mary. |
| Chci říct, že to není snadné. | I mean, it's not easy. | Now I want to say, this is not easy. |


## Technical Report

A detailed technical report describing the model architecture, design choices,
training setup, and experimental results is available here:

📄 **[Technical Report](docs/technical_report.pdf)**


## Installation

**Requirements:**

- Python 3.8+
- PyTorch 2.x
- HuggingFace Tokenizers
- Matplotlib

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

The project expects data in tab-separated format, with source and target sequences:
```
<target sequence> \t <source sequence>
```

Example `test_basic.txt` (data from [6]):
```
I see a new car.                	Vidím nové auto.
I see a small city.              	Vidím malé město.
I do not see the old library.    	Nevidím starou knihovnu.
```

**Split your dataset**

Using `split_dataset.py` you can split the file with pairs of sentences into `train.txt` and `test.txt`:
```
python split_dataset.py \
    --data_path data/raw.txt \
    --unshuffled \
    --range_train 0:1000 \
    --range_test 1000:1200 \
    --output_dir data/
```

## Training

Train the seq2seq Transformer using `train.py`:

```
python train.py \
    --data_path data/train.txt \
    --val_path data/val.txt \
    --tokenizer BPE \
    --use_rope \
    --max_vocab_size 10000 \
    --max_seq_len 64 \
    --d_model 256 \
    --num_heads 4 \
    --num_layers 4 \
    --d_ff 1024 \
    --dropout 0.1 \
    --epochs 20 \
    --lr 0.001 \
    --checkpoint_dir runs/
```

**Outputs**:

Model checkpoints, training loss plot and tokenizer states (`BPE` or `WORD` level) are saved in `runs/checkpoint_<timestamp>/`


## Evaluation

Evaluate a trained model on a test set:
```
python test.py \
    --data_path data/test.txt \
    --checkpoint_dir runs/checkpoint_20251120_123015 \
    --max_tgt_len 64
```

**Features**:

- Automatic source encoding with the stored tokenizer
- Autoregressive decoding using <SOS> and <EOS> tokens
- BLEU-N computation
- Prints predictions alongside references

**Example output**:

```
SRC:  Taková je bohužel realita, ve které musím žít.
PRED:  This is unfortunately the reality that I have to live in.
REF:  That’s the reality, unfortunately, in which I have to live.
BLEU-1: 0.690, BLEU-2: 0.480, BLEU-3: 0.349
```


## Project Structure

**Key modules**:

- `transformer.py` : Transformer encoder-decoder block and layer implementation
- `tokenizer.py` : Word-level and BPE tokenizer with save/load functions
- `dataset_utils.py` : Parsing and preprocessing datasets
- `utils.py` : Saving/loading model and tokenizer states, BLEU metric and helper functions


## References

[1] [*Vaswani et al., Attention Is All You Need (2017)*](https://arxiv.org/abs/1706.03762)

[2] [*Su et al., RoFormer: Enhanced Transformer with Rotary Position Embedding (2021)*](https://arxiv.org/abs/2104.09864)

[3] [*Ebad Sayed, Building a Transformer from Scratch: A Step-by-Step Guide (2024)*](https://medium.com/@sayedebad.777/building-a-transformer-from-scratch-a-step-by-step-guide-a3df0aeb7c9a)

[4] [*PyTorch*](https://pytorch.org/docs/stable/index.html)

[5] [*HuggingFace Tokenizers*](https://huggingface.co/docs/tokenizers/index)

[6] [*ManyThings.org: Tab-delimited Bilingual Sentence Pairs*](https://www.manythings.org/anki/)

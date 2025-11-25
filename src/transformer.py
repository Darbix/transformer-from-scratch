# ====================================================================
# File: transformer.py
# Author: David Kedra
#
# The code is based on the Transformer architecture from the paper:
# Attention Is All You Need (https://arxiv.org/abs/1706.03762)
# ====================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Add sinusoidal positional encodings to token embeddings."""

    def __init__(self, d_model, max_seq_length):
        """
        Args:
            max_seq_length: Maximum sequence length (number of positions).
        """
        super().__init__()

        # Tensor to hold positional encodings for all positions
        pe = torch.zeros(max_seq_length, d_model)
        # Position indices (0, 1, 2, ..., max_seq_length-1)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        # Sine/cosine frequency denominator
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        # div_term: 1D tensor of length d_model/2

        # Apply sine to even dimensions (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd dimensions (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Save as buffer so it’s part of the model but not a learnable parameter
        # Unsqueeze to add batch dimension: shape (1, max_seq_length, d_model)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        # Endpoint index if max_seq_length > seq_len
        seq_len = x.size(1)
        # Merge the embedding with the positional encoding
        return x + self.pe[:, :seq_len]


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    RoPE works by grouping adjacent dimensions within each token into d_model/2
    independent 2D pairs: (x_0, x_1), (x_2, x_3), ... Each such pair acts like
    a 2D point, representing one complex number (z_k = x_{2k} + i * x_{2k+1}).

    RoPE rotates each k-th feature pair of a token at position p by an angle
    θ_{p,k} = p * 10000^(-2k / d_model). Each token position is thus encoded
    as a rotation (angle), not as an additive bias to the embedding vector.

    The relative rotation between two positions directly encodes their distance.
    In the attention computation when two tokens (i, j) interact, the difference
    of embedded angles (θ_i - θ_j) reflects how far apart these tokens are.
    """

    def __init__(self, d_model):
        """
        Args:
            d_model (int): Embedding dimension (must be even).
        """
        super().__init__()
        assert d_model % 2 == 0, "RoPE requires even d_model."

        self.d_model = d_model

        # Precompute rotation angles
        pair_idcs = torch.arange(0, d_model // 2, dtype=torch.float)
        # thetas shape: (d_model//2,)
        theta_freqs = 10000 ** (-2 * pair_idcs / d_model)

        self.register_buffer("theta_freqs", theta_freqs)

    def forward(self, x):
        """
        Apply Rotary Position Embedding (RoPE) to tensor x.

        Args:
        x: Tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # Split last dim into 2D pairs
        x_pairs = x.view(batch_size, seq_len, d_model // 2, 2)
        # Shape: (batch_size, seq_len, d_model//2, 2)

        # token_positions shape: (seq_len,)
        token_positions = torch.arange(0, seq_len, device=x.device, dtype=torch.float)
        # Angles broadcasted for all tokens for all pair positions
        angles = token_positions[:, None] * self.theta_freqs[None, :]
        # angles shape: (seq_len, d_model//2)

        # Calculate sin and cos and expand for batch_dim: (1, seq_len, d_model//2)
        sin = torch.sin(angles).unsqueeze(0)
        cos = torch.cos(angles).unsqueeze(0)

        # Apply rotation to both first and second elements of pairs
        x1 = x_pairs[..., 0] # Shape: (batch, seq_len, d_model//2)
        x2 = x_pairs[..., 1] # Shape: (batch, seq_len, d_model//2)
        # Rotate and combine elements at the new last dimension
        x_rotated = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        # x_rotated shape: (batch, seq_len, d_model//2, 2)

        # Output is reshaped back to: (batch_size, seq_len, d_model)
        return x_rotated.view(batch_size, seq_len, d_model)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention block.

    Computes attention over the same sequence (self-attention) or between
    sequences (cross-attention) with multiple heads. Each head computes
    attention in its subspace: (batch_size, seq_len, d_k), and the outputs are
    concatenated back to the original embedding dimension via a linear layer
    as (batch_size, seq_len, num_heads * d_k) => (batch_size, seq_len, d_model).

    Args:
        d_model (int): Model embedding dimension.
        num_heads (int): Number of attention heads.
    """

    def __init__(self, d_model, num_heads, use_rope=False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."

        self.d_model = d_model          # Input token embedding dimension
        self.num_heads = num_heads      # Number of attention heads that divide input features between themselves
        self.d_k = d_model // num_heads # Dimension of Q, K, V matrices per head
        self.use_rope = use_rope
        # Positional embedding block if RoPE is selected
        self.rope = RotaryPositionalEmbedding(d_model) if use_rope else nn.Identity()

        # Linear projection weight matrices for queries, keys and values
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output linear projection
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Compute scaled dot-product attention for all heads.
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

        Args:
            Q: query vectors, shape: (batch_size, num_heads, tgt_seq_len, d_k)
            K:   key vectors, shape: (batch_size, num_heads, src_seq_len, d_k)
            V: value vectors, shape: (batch_size, num_heads, src_seq_len, d_k)
            mask: token mask of shape (batch_size, 1, 1 or seq_len, seq_len)

        Output shape: (batch_size, num_heads, tgt_seq_len, d_k)
        """
        # Compute scaled dot-product attention scores
        attn_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # attn_scores shape: (batch_size, num_heads, tgt_seq_len, src_seq_len)

        if mask is not None:
            # Replace invalid (padded/future) tokens with a huge negative number to be ignored
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Softmax across keys dimension (each row sums to 1)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        # attn_probs shape: (batch_size, num_heads, tgt_seq_len, src_seq_len)
        output = attn_probs @ V
        # output shape: (batch_size, num_heads, tgt_seq_len, d_k)
        return output

    def split_heads(self, x):
        """
        Split the embedding dimension by reshaping it into multiple subspaces.
        Each head views different d_k dimensions of the input embedding d_model.

        Args:
            x: Input of shape (batch_size, seq_len, d_model)
        Returns:
            Output of shape (batch_size, num_heads, seq_len, d_k), where d_k = d_model // num_heads
        """
        # x shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape

        x = x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # x shape: (batch_size, num_heads, seq_len, d_k)
        return x

    def forward(self, Xq, Xk, Xv, mask=None):
        """
        Compute multi-head attention over the input sequence.

        Args:
            Xq: query vectors, shape: (batch_size, tgt_seq_len, d_model)
            Xk:   key vectors, shape: (batch_size, src_seq_len, d_model)
            Xv: value vectors, shape: (batch_size, src_seq_len, d_model)
            mask: Broadcastable mask for padded or future token blocking.
                  Shape (batch_size, 1, 1 or tgt_seq_len, src_seq_len).
        Notes:
            Xq, Xk, Xv: Input vectors of shape (batch_size, seq_len, d_model)
                - In self-attention: Xq = Xk = Xv => tgt_seq_len = src_seq_len
                - In cross-attention: Xq comes from the decoder,
                                      Xk and Xv come from the encoder output.
        Returns:
            Output of shape (batch_size, tgt_seq_len, d_model)
        """
        batch_size = Xq.shape[0]

        # Linear projections of input by weight matrices (keeps shape)
        Q = self.W_q(Xq)
        K = self.W_k(Xk)
        V = self.W_v(Xv)

        # Apply RoPE to Q and K if enabled
        if self.use_rope:
            # Q = apply_RoPE(Q)
            # K = apply_RoPE(K)
            Q = self.rope(Q)
            K = self.rope(K)

        # Reshape the matrices into multiple attention heads
        Q = self.split_heads(Q) # Shape: (batch_size, num_heads, tgt_seq_len, d_k)
        K = self.split_heads(K) # Shape: (batch_size, num_heads, src_seq_len, d_k)
        V = self.split_heads(V) # Shape: (batch_size, num_heads, src_seq_len, d_k)

        # Compute scaled dot-product attention for each head
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        # attn_output shape: (batch_size, num_heads, tgt_seq_len, d_k)

        # Combine all heads back into one sequence representation
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous() # Ensures tensor is continuous in memory after transpose
            .view(batch_size, -1, self.d_model)
        )
        # attn_output shape: (batch_size, tgt_seq_len, d_model)

        # Output linear projection
        output = self.W_o(attn_output)
        # output shape: (batch_size, tgt_seq_len, d_model)
        return output


class PositionwiseFeedForward(nn.Module):
    """Feed-forward block applied independently to each element in the sequence."""

    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff)
        x = self.fc1(x)
        x = F.relu(x)
        # (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    """Encoder layer of the Transformer model."""

    def __init__(self, num_heads, d_model, d_ff, dropout, use_rope=False):
        super().__init__()

        # Self-attention block
        self.self_attn = MultiHeadAttention(d_model, num_heads, use_rope=use_rope)
        self.norm1 = nn.LayerNorm(d_model) # Post-Norm (after ...)

        # Feed-forward block
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x & output shape: (batch_size, seq_len, d_model)

        # Multi-head self attention
        # Pre-Norm normalizes before feeding the sublayer
        # (Post-Norm would normalize after adding the residual)
        x_norm = self.norm1(x)
        attn_output = self.self_attn(x_norm, x_norm, x_norm, mask)
        x = x + self.dropout(attn_output)

        # Feed-forward
        x_norm = self.norm2(x)
        ff_output = self.feed_forward(x_norm)
        x = x + self.dropout(ff_output)
        return x


class DecoderLayer(nn.Module):
    """Decoder layer of the Transformer model."""

    def __init__(self, d_model, num_heads, d_ff, dropout, use_rope=False):
        super().__init__()

        # Masked self-attention block
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads, use_rope=use_rope)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention block
        self.cross_attn = MultiHeadAttention(d_model, num_heads, use_rope=use_rope)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward block
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask, src_mask):
        """
        Args:
            x: Input of shape (batch_size, tgt_seq_len, d_model)
            enc_output: Encoder output of shape (batch_size, src_seq_len, d_model)
            src_mask: Prevents the attending to padded token positions
            tgt_mask: Prevents a token from attending to future tokens
        Returns:
            Output of shape (batch_size, tgt_seq_len, d_model)
        """
        # Masked self-attention
        # Pre-Norm normalizes before feeding the sublayer
        # (Post-Norm would normalize after adding the residual)
        x_norm = self.norm1(x)
        masked_attn_output = self.masked_self_attn(x_norm, x_norm, x_norm, tgt_mask)
        x = x + self.dropout(masked_attn_output)

        # Cross-attention
        # enc_output is already stable and does not need normalization
        x_norm = self.norm2(x)
        cross_attn_output = self.cross_attn(x_norm, enc_output, enc_output, src_mask)
        x = x + self.dropout(cross_attn_output)

        # Feed-forward
        x_norm = self.norm3(x)
        ff_output = self.feed_forward(x_norm)
        x = x + self.dropout(ff_output)
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048,
                 dropout=0.1, max_seq_length=512, use_rope=True):
        """
        A Transformer model with the standard encoder-decoder architecture
        as described in the paper "Attention Is All You Need".

        Args:
            src_vocab_size (int): Size of the source vocabulary.
            tgt_vocab_size (int): Size of the target vocabulary.
            d_model (int): The embedding dimension.
            num_heads (int): Number of attention heads in the Multi-Head Attention layer.
            num_encoder_layers (int): Number of encoder layers.
            num_decoder_layers (int): Number of decoder layers.
            d_ff (int): Dimension of the feedforward network model.
            dropout (float): The dropout probability.
            max_seq_length (int): Maximum sequence length for positional encoding.
        """
        super().__init__()

        # Token embeddings + positional encoding
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.use_rope = use_rope
        # Switch between absolute sinusoidal positional embedding and RoPE
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length) if not use_rope else nn.Identity()

        # Stack of encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(num_heads, d_model, d_ff, dropout, use_rope=use_rope)
            for _ in range(num_encoder_layers)
        ])

        # Stack of decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, use_rope=use_rope)
            for _ in range(num_decoder_layers)
        ])

        # Fully connected layer that maps d_model to vocab logits
        # Each output neuron corresponds to one vocabulary token
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def create_padding_mask(self, seq, pad_token_id=0):
        """
        Create a mask for padding tokens so the model ignores them in attention.

        Args:
            seq: Input sequence of shape (batch_size, seq_len)
        Returns:
            Output of shape (batch_size, 1, 1, seq_len) for broadcasting across heads & queries
        """
        return (seq != pad_token_id).unsqueeze(1).unsqueeze(2)

    def create_look_ahead_mask(self, seq_len):
        """
        Create a mask to block attention to future positions.

        Output: lower-triangular matrix of shape (1, 1, seq_len, seq_len)

        Mask visualization over queries & keys positions:
        Tokens: tK1 tK2 tK3
            tQ1 [1,  0,  0]
            tQ2 [1,  1,  0]
            tQ3 [1,  1,  1]
        """
        return torch.tril(torch.ones((seq_len, seq_len))).unsqueeze(0).unsqueeze(0)

    def encode(self, src, src_mask):
        """Encoder transformer part."""
        # Embed input token indices (source sentences) into token embeddings
        # src shape: (batch_size, src_seq_len)
        x = self.src_embedding(src)
        # x shape: (batch_size, src_seq_len, d_model)

        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        # Output x shape: (batch_size, src_seq_len, d_model)
        return x

    def decode(self, tgt, enc_output, tgt_mask, src_mask):
        """Decoder transformer part."""
        # Embed input token indices (target sentences) into token embeddings
        # tgt shape: (batch_size, tgt_seq_len)
        x = self.tgt_embedding(tgt)
        # x shape: (batch_size, tgt_seq_len, d_model)

        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.decoder_layers:
            x = layer(x, enc_output, tgt_mask, src_mask)
        # Output x shape: (batch_size, tgt_seq_len, d_model)
        return x

    def forward(self, src, tgt):
        """
        Args:
            src: Source input sequence of shape (batch_size, src_seq_len)
            tgt: Target input sequence of shape (batch_size, tgt_seq_len)
        Returns:
            Output logits of shape (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # Hide padding tokens from src sequence
        src_mask = self.create_padding_mask(src)
        # src_mask shape: (batch_size, 1, 1, src_seq_len)

        # Hide padding + future tokens from tgt sequence
        tgt_pad_mask = self.create_padding_mask(tgt)
        tgt_look_ahead_mask = self.create_look_ahead_mask(tgt.size(1)).to(tgt.device).bool()
        tgt_mask = tgt_pad_mask & tgt_look_ahead_mask
        # tgt_mask shape: (batch_size, 1, tgt_seq_len, tgt_seq_len)

        enc_output = self.encode(src, src_mask)
        # enc_output shape: (batch_size, src_seq_len, d_model)

        dec_output = self.decode(tgt, enc_output, tgt_mask, src_mask)
        # enc_output shape: (batch_size, tgt_seq_len, d_model)

        logits = self.fc(dec_output)
        # Output logits shape: (batch_size, tgt_seq_len, tgt_vocab_size)
        return logits

    def inference(self, src, sos_id, eos_id, max_tgt_len=64):
        """
        Autoregressive inference (greedy decoding).

        Args:
            src: Tensor of shape (1, src_seq_len)
            sos_id: <SOS> token ID
            eos_id: <EOS> token ID
        Returns:
            A list of predicted token IDs (without <SOS>)
        """
        self.eval()

        # Create encoder mask and encode source sentence only once
        src_mask = self.create_padding_mask(src)
        enc_output = self.encode(src, src_mask)

        # Start with <SOS> token and add the batch dimenstion
        tgt = torch.tensor([[sos_id]], device=src.device)

        for _ in range(max_tgt_len):
            # Build look-ahead mask for current partial tgt (no padding needed)
            tgt_mask = self.create_look_ahead_mask(tgt.size(1)).bool().to(src.device)

            # Decode using current generated prefix
            dec_output = self.decode(tgt, enc_output, tgt_mask, src_mask)

            # Predict next token using the last generated token from the output
            # of shape: (batch_size, tgt_seq_len, tgt_vocab_size)
            logits = self.fc(dec_output[:, -1, :])
            next_token = torch.argmax(logits, dim=-1).item()

            if next_token == eos_id:
                break

            # Append next token to an autoregressive input sequence
            next_token_tensor = torch.tensor([[next_token]], device=src.device)
            tgt = torch.cat([tgt, next_token_tensor], dim=1)

        # Remove <SOS> and return the predicted sequence
        return tgt[0].tolist()[1:]


#!/home/akugyo/Programs/Python/torch/bin/python

import torch
import math
from torch import nn


class InputEmbeddings(nn.Module):
    """
    Convert input tokens into a vector of size 512 as mentioned in the
    paper.
    To test this create a dictionary that maps words with numbers,
    convert the words to numbers using the dict and and convert the
    resultant to tensor which becomes the x.
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
        Takes 2 arguments: d_model(int), vocab_size(int), where d_model
        is the size of the vector whose value given in the paper is 512,
        vocab_size defines the size of the vocabulary.
        """

        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor):
        """
        The embedding is multiplied by a factor of sqrt of the d_model
        as specified by the paper.
        """

        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Given a set of input tokens, add positional encodings to it and
    return the output. Created only once, not a parameter.
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        """
        Takes 3 arguments: d_model(int), vocab_size(int), dropout(int.)
        """

        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a tensor of size seq_len * d_model for the positional
        # encoding
        positional_encoding = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, stype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        # Apply sin to even positions and cos to odd positions
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        # Increase the dimension by 1 because data is sent in the form
        # of batches (1, seq_len, d_model)
        positional_encoding = positional_encoding.unsqueeze(0)
        # Stores pe in model but not as a parameter, but allows to save
        # it when saving a model
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to Input Embedding
        """

        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    """
    Performs Layer Normalization
    """

    def __init__(self, eps: float = 10**-6):
        """
        Takes 1 argument: eps.
        """

        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor):
        """
        (Data point - mean) / (standard_deviation + epsilon) * alpha
        + bias
        """

        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    """
    Contains of 2 linear layers, a relu and a dropout in between them.
    """

    def __init__(self, d_model: int, linear_middle: int, dropout: float):
        """
        Takes 3 arguments: d_model(int), linear_middle(int),
        dropout(int).
        """

        super().__init__()
        self.feed_forward = nn.Sequential(nn.Linear(d_model, linear_middle),
                                          nn.ReLU(),
                                          nn.Dropout(),
                                          nn.Linear(linear_middle, d_model),
        )

    def forward(self, x: torch.Tensor):
        return self.feed_forward(x)


class MultiHeadAttentionBlock(nn.Module):
    """
    Defines multi head attention.
    Create query, key, value.
    Pass them into their respctive linear layers.
    Change the view to split the matrices into smaller matrices.
    Apply attention.
    Pass it to linear and send it out.
    """

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        """
        Takes 2 arguments: d_model(int), h(int) number of heads,
        dropout(float)
        """

        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h ==0, "d_model not divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = dropout

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """
        (Query * key) / root(d_k).
        Apply mask if available.
        Apply softmax.
        Apply dropout if available.
        """

        d_k = query.shape[-1]
        # (batch, h, seq_le, d_k) -> (batch, h, seq_le, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ values), attention_scores

    def forward(self, q, k ,v, mask):
        """
        Apply linear layers for k, q, v.
        Apply attention.
        Apply linear layer after attention and send output.
        """

        # Apply linear layers
        query = self.w_k(q)
        key = self.w_k(k)
        values = self.w_v(v)

        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) ->
        # (batch, h, seq_len, d_k)
        # Split matrics into smaller matrices
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k
                           ).transpose(1, 2)
        key = key.view(kew.shape[0], key.shape[1], self.h, self.d_k
                       ).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k
                           ).transpose(1, 2)

        # Apply attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout)
        # (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k) ->
        # (batch, seq_lem, d_model)
        # # Change shape to pass it to the final layer
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h
                                                * self.d_k)

        return self.w_o(x)


class ResidualConnection(nn.Module):
    """
    Defines the skip orward connection and also does Add&Norm.
    """

    def __init__(self, dropout: float) -> None:
        """
        Takes 1 argument: dropout(int).
        """

        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        """
        Apply normalization, sublayer, dropout and add.
        """

        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    """
    Create all subblock required to create a Encoder Block.
    """

    def __init__(self, self_attention_bloack: MultiHeadAttentionBloack,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        """
        Takes 3 arguments: self_attention_block, feed_forward_block, dropout.
        """

        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidulaConnection(dropout)
                                                   for _ in range(2)])

    def forward(self, x, src_mask):
        """
        Does mutltiheadateention and Add&Norm.
        """

        x = self.residual_connections[0](x, lambda x:
                                         self.self_attention_block(
                                             x, x, x, src_mask)
                                         )
        x = self.residual_connections[1](x, self.feed_forward_block)

        return x


class Encoder(nn.Module):
    """
    Encoder block
    """

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):

        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

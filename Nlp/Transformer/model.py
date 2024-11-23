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

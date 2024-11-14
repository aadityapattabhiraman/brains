#!/home/akugyo/Programs/Python/torch/bin/python


import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import Dataset
from torch.utils.tensorboard import SummaryWriter


class CharDataset(Dataset):

    def __init__(self, words, chars, max_word_length):
        self.words = words
        self.chars = chars
        self.max_word_length = max_word_length
        self.stoi = {ch: i+1 for i, ch in enumerate(chars)}
        self.itos = {i: s for s, i in self.stoi.items()}

    def __len__(self):
        return len(self.words)

    def contains(self, word):
        return word in self.words

    def get_vocab_size(self):
        return len(self.chars) + 1

    def get_output_length(self):
        return self.max_word_length + 1

    def encode(self, word):
        return torch.tensor([self.stoi[w] for w in word], dtype=torch.long)

    def decode(self, ix):
        return "".join(self.itos[i] for i in ix)

    def __getitem__(self, idx):
        word = self.words[idx]
        ix = self.encode(word)
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        x[1: 1+len(ix)] = ix
        y[: len(ix)] = ix
        y[len(ix)-1: ] = -1
        return x, y

class Bigram(nn.Module):

    def __init__(self, config):
        super().__init__()
        n = config.vocab_size
        self.logits = nn.Parameter(torch.zeros((n, n)))

    def get_block_size(self):
        return 1

    def forward(self, idx, targets=None):
        logits = self.logits[idx]
        loss = None

        if targets is not None:
            loss = nn.CrossEntropyLoss(logits.vire(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1)

        return logits, loss

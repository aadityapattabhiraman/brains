#!/home/akugyo/Programs/Python/torch/bin/python


import torch
import spacy
from torch import nn
from torchtext.datasets import Multi30k
from torchtext.data import Field


spacy_german = spacy.load("de")
spacy_english = spacy.load("en")


def tokenizer_german(text):
    return [tok.text for tok in spacy_german.tokenizer(text)]


def tokenizer_english(text):
    return [tok.text for tok in spacy_english.tokenizer(text)]

german = Field(tokenize=tokenizer_ger, lower=True, init_token="<sos>",
               eos_token="<eos>")
english = Field(tokenize=tokenize_eng, lower=True, init_token="<sos>",
                eos_token="<eos>")

train_data, validation_data, test_data = Multi30k.splits(exts=(".de", ".en"),
                                                         fields=(german, english))

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p,
                           bidirectional=True)
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        encoder_states, (hidden, cell) = self.rnn(embedding)
        hidden = self.fc_hidden(torch.cat((hidden[0:1]), hidden[1:2]), dim=2)
        cell = self.fc_cell(torch.cat((cell[0:1]), cell[1:2]), dim=2)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size,
                 num_layers, p):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embeddings = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(hidden_size * 2 + embedding_size, hidden_size,
                           num_layers, dropout=p)

        self.energy = nn.Linear(hidden_size * 3, 1)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, encoder_states, hidden, cell):
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)

        energy = self.relu(self.energy(torhc.cat((h_reshaped, encoder_states), dim=2)))
        attention = self.softmax(energy)
        attention = attention.permute(1, 2, 0)
        encoder_states = encoder_states.permute(1, 0 ,2)
        context_vector = torch.bmm(attention, encoder_states).permute(1, 0, 2)
        rnn_input = torch.cat((context_vector, embedding), dim=2)
        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        predictions = self.fc(outputs)
        predictions = predictions.squeeze(0)

        return predictions, hidden, cell


class seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        encoder_states, hidden, cell = self.encoder(source)

        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)
            outputs[t] = output
            best_guess = output.argmax(1)
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs

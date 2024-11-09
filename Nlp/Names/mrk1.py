#!/home/akugyo/Programs/Python/torch/bin/python

import torch
import matplotlib.pyplot as plt


words = open("../../Dataset/names.txt", "r").read().splitlines()
## Analyse the dataset ##
# print(words[:10])
# print(len(words))

## Dictionary for the set of data ##
# b = {}
# for w in words:
# chs = ["<S>"] + list(w) + ["<E>"]
# for ch1, ch2 in zip(chs, chs[1:]):
# print((ch1, ch2))
# bigram = (ch1, ch2)
# b[bigram] = b.get(bigram, 0) + 1

# print(sorted(b.items(), key=lambda kv: -kv[1]))

## Dictionaries are inefficient hence convert to tensors ##
N = torch.zeros((53, 53), dtype=torch.int32)
chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["<>"] = 0

itos = {s: i for i, s in stoi.items()}

for w in words:
    chs = ["<>"] + list(w) + ["<>"]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

# print(N)

## Calculate for every dimension in while ##
## Use P later ##
P = (N+1).float()
P /= P.sum(1, keepdim=True)

for i in range(20):
    ix = 0
    out = []
    while True:
        p = P[ix]
        # print(p)
        ix = torch.multinomial(p, num_samples=1, replacement=True).item()
        # print(itos[ix])
        out.append(itos[ix])
        if ix == 0:
            break
    print("".join(out))

## Calculate loss ##
log_likelihood = 0.0
n = 0
for w in words:
    chs = ["<>"] + list(w) + ["<>"]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        log_prob = torch.log(prob)
        log_likelihood += log_prob
        n += 1
        # print(f"{ch1}{ch2}: \t{prob:.4f} {log_prob:.4f}")

negative_log_likelihood = - log_likelihood
print(f"{negative_log_likelihood=}")
print(f"{negative_log_likelihood/n}")

# ["andrejq"] example add 1 model smoothing

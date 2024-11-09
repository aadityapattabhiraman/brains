#!/home/akugyo/Programs/Python/torch/bin/python

import torch
from torch.nn import functional as F
from time import time


start = time()
words = open("../../Dataset/names.txt", "r").read().splitlines()

chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["<>"] = 0
itos = {s: i for i, s in stoi.items()}

xs, ys = [], []
for w in words[: 1]:
    chs = ["<>"] + list(w) + ["<>"]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

x_encoded = F.one_hot(xs, num_classes=53).float()
print(x_encoded.shape)
print(x_encoded.dtype)

## Forward Pass ##
W = torch.randn((53, 53), requires_grad=True)
logits = x_encoded @ W
counts = logits.exp()
prob = counts / counts.sum(1, keepdim=True)
## Last 2 lines are called softmax just call nn.Softmax() ##
## Loss ##
loss = -prob[torch.arange(10), ys].log().mean()
W.grad = None
## Back Propagate ##
loss.backward()
W.data == -0.1 * W.grad

xs, ys = [], []
for w in words:
    chs = ["<>"] + list(w) + ["<>"]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print("Number of examples:", num)

W = torch.randn((53, 53), requires_grad=True)

## Training Loop ##
for k in range(100):
    x_encoded = F.one_hot(xs, num_classes=53).float()
    logits = x_encoded @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W**2).mean()
    # print(loss.item())

    W.grad = None
    loss.backward()

    W.data += -10 * W.grad
print(loss.item())
end = time()
print(f"Time Taken: {end - start:.4f} seconds")

## Inference Loop ##
for i in range(5):
    out = []
    ix = 0
    while True:
        x_encoded = F.one_hot(torch.tensor([ix]), num_classes=53).float()
        logits = x_encoded @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdims=True)

        ix = torch.multinomial(p, num_samples=1, replacement=True).item()
        out.append(itos[ix])

        if ix == 0:
            break
    print("".join(out))

# Dependencies
import numpy as np
import os, random, pickle
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from itertools import permutations
from torch.utils.data import DataLoader, Dataset

from model import (
    GPT,
    encode,
    decode,
    create_dataset,
    CustomDataset,
    CausalSelfAttention,
    LayerNorm,
    MLP,
    Block,
    training_loop,
)

from config import GPTConfig


def make_data1():
    if not os.path.isfile("./shakespeare.txt"):
        os.system(
            "wget https://raw.githubusercontent.com/tikendraw/infinite-shakespeare/main/shakespeare.txt"
        )

    with open("shakespeare.txt", "r") as f:
        text = f.readlines()

    text = " ".join(text)

    chars = "".join(sorted(set(text)))

    # # Tokenization

    # We will create 2 character level token
    a = permutations(chars, 2)
    a = list(a)

    # permuation does not include chars like aa, bb 99 , doin that
    a.extend("".join(i) for i in zip(chars, chars))
    # joining tuples
    a = ["".join(i) for i in a]  # 7140 combination possible

    # keeping only tokens that exist in data
    a = [i for i in a if i in text]  # 2018 combination exists in data

    stoi = {i: num for num, i in enumerate(a, 1)}  # token
    stoi["UNK"] = 0
    itos = {i: num for num, i in stoi.items()}
    return text, stoi, itos


import contextlib

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on Device: ", device)
    print("Preprocessing...")
    # Config
    config = GPTConfig(device=device)

    VOCAB_SIZE = config.vocab_size
    RESERVED_TOKEN = 0  # for unknown token

    text, stoi, itos = make_data1()

    enc_text = encode(text, stoi)
    x, y = create_dataset(enc_text, config.block_size)

    # DataLoader
    dataset = CustomDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    gpt = GPT(config).to(config.device)
    gpt.itos = itos
    gpt.stoi = stoi

    # loading weights if exist
    if os.path.isfile("/model_weights/gpt.pth"):
        try:
            gpt.load_state_dict(torch.load("/model_weights/gpt.pth"))
        except:
            print("loading weights failed! ")
    # Training
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=config.learning_rate)

    epochs = config.epochs

    print("Training..")
    for epoch in range(1, epochs + 1):
        epoch_loss = training_loop(
            gpt.to(config.device),
            dataloader,
            optimizer=optimizer,
            train_step=10,
            device=config.device,
        )

        if epoch % epochs // 11 == 0:
            print(f"Epoch:  {epoch:4}/{epochs}  |  Loss:  {epoch_loss:.5f}")

    print("Training Done!")

    os.makedirs("model_weights", exist_ok=True)

    torch.save(gpt.state_dict(), "./model_weights/gpt.pth")
    print("Model saved!")

import os
import torch
import numpy as np
from model import Model
from torch.optim import AdamW
from data.tokenizer import Tokenizer
from config import Config

raw_dataset_dir = "data/datasets/text.txt"
bin_training_dataset_dir = "data/dataset/train.bin"
bin_validation_dataset_dir = "data/dataset/validate.bin"
merges_dir = "data/datasets/merges.json"


def tokenize_dataset(dir, tokenizer):
    with open(dir, "r") as f:
        raw_text = f.read()

    ids = tokenizer.encode(raw_text)

    print(f"{len(raw_text)} characters to {len(ids)} tokens")

    split = int(len(ids) * 0.8)

    np.array(ids[:split], dtype=np.uint16).tofile(bin_training_dataset_dir)
    np.array(ids[split:], dtype=np.uint16).tofile(bin_validation_dataset_dir)
    

def get_batch(dir):

    dataset = np.fromfile(dir, dtype=np.uint16)
    batch_size = Config.batch_size
    block_size = Config.block_size

    starting_index = torch.randint(0, len(dataset) - block_size, (batch_size, ))

    x = torch.stack([torch.from_numpy(dataset[i: i + block_size].astype(np.int64)) for i in starting_index]).to("cuda")
    y = torch.stack([torch.from_numpy(dataset[i + 1:i + block_size + 1].astype(np.int64)) for i in starting_index]).to("cuda")

    return x, y


if __name__ == "__main__":
    print('start training')

    tokenizer = Tokenizer()
    tokenizer.load(merges_dir)

    if not os.path.isfile(bin_training_dataset_dir):
        tokenize_dataset(tokenizer)


    model = Model(Config()).to("cuda")

    optimizer = AdamW(params=model.parameters(),lr = 3e-4)

    for epoch in range(5000):
        x, y = get_batch(bin_training_dataset_dir)

        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            print(f"epoch {epoch + 1} | loss {loss.item()}")
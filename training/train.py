import os
import math
import torch
import numpy as np
from model import Model
from config import Config
from torch.optim import AdamW
from dataclasses import asdict
from data.tokenizer import Tokenizer

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

@torch.no_grad()
def evaluate_loss(model):
    out = {}
    model.eval()
    for dataset, dataset_dir in {
        "validation" : bin_validation_dataset_dir,
        "training": bin_training_dataset_dir
    }.items():
        losses = torch.zeros(Config.evaluation_epochs)
        for i in  range(Config.evaluation_epochs):
            x, y = get_batch(dataset_dir)
            logits, loss = model(x, y)
            losses[i] = loss.item()
        out[dataset] = losses.mean()

    model.train()
    return out


def get_batch(dir):

    dataset = np.fromfile(dir, dtype=np.uint16)
    batch_size = Config.batch_size
    block_size = Config.block_size

    starting_index = torch.randint(0, len(dataset) - block_size, (batch_size, ))

    x = torch.stack([torch.from_numpy(dataset[i: i + block_size].astype(np.int64)) for i in starting_index]).to("cuda")
    y = torch.stack([torch.from_numpy(dataset[i + 1:i + block_size + 1].astype(np.int64)) for i in starting_index]).to("cuda")

    return x, y

def configure_optimizer(model):
    # we only want to add weight decay to the weights not the biases
    to_decay = [p for p in model.parameters() if p.dim() >= 2]
    to_not_decay = [p for p in model.parameters() if p.dim() < 2]

    optimizer = torch.optim.AdamW([
        {'params': to_decay,  "weight_decay" : Config.weight_decay},
        {"params" : to_not_decay, "weight_decay": 0}
    ])

    return optimizer

def get_learning_rate(epoch):
    warmup_epochs, decay_epochs, learning_rate = Config.learning_rate_warmup_epochs, Config.learning_rate_decay_epochs, Config.learning_rate
    if epoch < warmup_epochs:
        return learning_rate * ((epoch + 1 ) / (warmup_epochs + 1))
    elif epoch > decay_epochs:
        return learning_rate
    else:
        # cosine decay
        decay_ratio = (epoch - warmup_epochs ) / (decay_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * decay_ratio))

if __name__ == "__main__":
    print('start training')

    tokenizer = Tokenizer()
    tokenizer.load(merges_dir)

    if not os.path.isfile(bin_training_dataset_dir):
        tokenize_dataset(tokenizer)


    model = Model(Config()).to("cuda")

    optimizer = configure_optimizer(model)


    best_loss = float("inf")

    for epoch in range(Config.training_epochs):
        current_learning_rate = get_learning_rate(epoch)

        for g in optimizer.param_groups:
            g['lr'] = current_learning_rate



        # gradient accumulation so the GPUs dont explode
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(Config.accumulation_steps):
            x, y = get_batch(bin_training_dataset_dir)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, loss = model(x, y)
                loss = loss / Config.accumulation_steps
            loss.backward()

        if Config.grad_clip != 0:
            # gradient clipping, so one bad run doesnt throw off all the weights
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.grad_clip)

        optimizer.step()

        if loss.item() < best_loss:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': asdict(Config()),   
                'epoch': epoch,
                'min_loss': min,
            }, 'ckpt.pt')


        if (epoch + 1) % 1000 == 0:
            print(f"epoch {epoch + 1} | loss {loss.item()}")
import os
import math
import torch
import argparse
import numpy as np
from model import Model
from config import Config
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.setup import ddp
from dataclasses import asdict
from data.tokenizer import Tokenizer

raw_dataset_dir = "data/datasets/text.txt"
bin_training_dataset_dir = "data/datasets/train.bin"
bin_validation_dataset_dir = "data/datasets/validate.bin"
merges_dir = "data/merges.json"

parser = argparse.ArgumentParser()
parser.add_argument("--resume_from_checkpoint", type=bool, default=False, required=False)
# can only use fp8 precision on H100 GPUs or later
parser.add_argument("--use-fp8", type=bool, default=False, required=False)

args = parser.parse_args()


can_use_ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = ddp() 
master_process = ddp_rank == 0

model = Model(Config()).to(device)
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module

def tokenize_dataset(dir, tokensizer):
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

    x = torch.stack([torch.from_numpy(dataset[i: i + block_size].astype(np.int64)) for i in starting_index]).to(device)
    y = torch.stack([torch.from_numpy(dataset[i + 1:i + block_size + 1].astype(np.int64)) for i in starting_index]).to(device)

    return x, y

def configure_optimizer(model):
    # we only want to add weight decay to the weights not the biases
    to_decay = [p for p in model.parameters() if p.dim() >= 2]
    to_not_decay = [p for p in model.parameters() if p.dim() < 2]

    optimizer = torch.optim.AdamW([
        {'params': to_decay,  "weight_decay" : Config.weight_decay},
        {"params" : to_not_decay, "weight_decay": 0}
    ], lr=Config.learning_rate, betas=(0.9, 0.95), fused=True)

    return optimizer

def get_learning_rate(epoch):
    warmup_epochs, decay_epochs, learning_rate, minimum_learning_rate = Config.learning_rate_warmup_epochs, Config.learning_rate_decay_epochs, Config.learning_rate, Config.minimum_learning_rate
    if epoch < warmup_epochs:
        return learning_rate * ((epoch + 1 ) / (warmup_epochs + 1))
    elif epoch > decay_epochs:
        return minimum_learning_rate
    else:
        # cosine decay
        decay_ratio = (epoch - warmup_epochs ) / (decay_epochs - warmup_epochs)
        coefficient = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return minimum_learning_rate + coefficient * (learning_rate - minimum_learning_rate)

if __name__ == "__main__":
    print('start training')

    tokenizer = Tokenizer()
    tokenizer.load(merges_dir)

    if not os.path.isfile(bin_training_dataset_dir):
        tokenize_dataset(merges_dir, tokenizer)


    optimizer = configure_optimizer(model)

    if args.resume_from_checkpoint and  os.path.isfile("checkpoint.pt"):
        checkpoint = torch.load("checkpoint.pt")
        model.load_state_dict(checkpoint['model'], strict=True, assign=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        starting_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['min_loss']
    else:
        best_loss = float("inf")
        starting_epoch = 0


    for epoch in range(starting_epoch, Config.training_epochs):
        print(f"epoch {epoch}")
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


        # gradient checkpointing
        if loss.item() < best_loss:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': asdict(Config()),   
                'epoch': epoch,
                'min_loss': best_loss,
            }, 'checkpoint.pt')

        best_loss = loss.item()

        if (epoch + 1) % Config.evaluation_epochs == 0:
            losses = evaluate_loss(model)
            print(f"epoch {epoch} | train {losses['training']:.4f} | val {losses['validation']:.4f}")
        

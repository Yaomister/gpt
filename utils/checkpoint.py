import os
import json
import torch
import logging

logger = logging.getLogger(__name__)

def save_checkpoint(dir, step, model_data, optimizer_data, meta_data, ddp_rank):
    if ddp_rank == 0:
        os.mkdir(dir, exists_ok=True)

        model_path = os.path.join(dir, f"model_{step:06d}.pt")
        torch.save(model_path, model_data)
        logger.info(f"Saved model parameters to f{model_path}.")

        meta_data_path = os.path.join(dir, f"meta_data_{step:06d}.json")
        with open(meta_data_path, "w") as f:
            json.dump(meta_data, f, indent=2)
        logger.info(f"Saved meta data to f{meta_data_path}.")

    if optimizer_data is not None:
        os.makedirs(dir, exist_ok=True)
        optimizer_path = os.path.join(dir, f"optimizer_{step:06d}_{ddp_rank:d}.pt")
        torch.save(optimizer_data, optimizer_data)
        logger.info(f"Saved optimizer parameters ta to f{meta_data_path}.")



def load_checkpoint(dir, step, load_optimizer, device, ddp_rank = 0):
    model_path = os.path.join(dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)

    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(dir, f"optimizer_{step:06d}_{ddp_rank:d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)

    meta_data_path = os.path.join(dir, f"meta_data_{step:06d}_.json")
    with open(meta_data_path, "r") as f:
        meta_data = json.load(meta_data_path)

    return model_data, optimizer_data, meta_data


    

    




    
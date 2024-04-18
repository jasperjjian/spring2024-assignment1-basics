from collections.abc import Callable, Iterable
from typing import Optional
import torch
import torch.nn as nn
import numpy.typing as npt
import numpy as np

def dataloader(dataset, batch_size, context_length, device):
    sample = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    sample_indices = np.array([np.arange(x, x+context_length) for x in sample])
    offset_indices = np.array([np.arange(x+1, x+context_length+1) for x in sample])

    sampled_dataset = np.take(dataset, sample_indices).astype(np.int16)
    sampled_offset = np.take(dataset, offset_indices).astype(np.int16)
    
    sampled_dataset = torch.tensor(sampled_dataset, dtype=torch.long).to(device)
    sampled_offset = torch.tensor(sampled_offset, dtype=torch.long).to(device)

    return sampled_dataset, sampled_offset

def save_checkpoint(model, optimizer, iteration, out):
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    checkpoint = {"model" : model_state, "optimizer" : optimizer_state, "iteration" : iteration}
    torch.save(checkpoint, out)
    return

def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    iteration = checkpoint["iteration"]
    return iteration
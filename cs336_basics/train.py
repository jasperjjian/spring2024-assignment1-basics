from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
import torch.nn as nn
import wandb
import transformer, optimizer, loss, utils
from tqdm import tqdm
import time

def train(model, train_loader, valid_loader, optim, criterion, steps, device, checkpoints=None):
    config = wandb.config
    #learning_rate = config.learning_rate
    validation_interval = 100
    model.to(device)
    model.train()
    train_loss = 0.0
    start_time = time.time()
    for step, (inputs, targets) in enumerate(train_loader, start=1):
        if step > steps:
            break
        if step % 2500 == 0 and checkpoints != None:
            print(step)
            path = checkpoints + f"model_{str(config.learning_rate)}_{step}"
            utils.save_checkpoint(model, optim, step, path)

        optim.zero_grad()
        preds = model(inputs)
        loss = criterion(preds, targets)
        loss.backward()
        """cosine_lr = optimizer.AdamW.cosine_schedule(step, config.learning_rate, 1e-3, steps//20, steps)
        for group in optim.param_groups:
            group["lr"] = cosine_lr"""
        train_loss = optim.step()
        wandb.log({'loss': loss.item(), 'step': step})

        if step % validation_interval == 0:
            model.eval()
            valid_len = 100
            v_loss = 0
            for i, (input, target) in enumerate(valid_loader):
                if i == valid_len:
                    break
                pred = model(input)
                v_loss += criterion(pred, target).item()
            v_loss /= valid_len
            curr = time.time()
            wandb.log({'valid_loss': v_loss, 'step': step, 'wall_clock' : curr-start_time})
            model.train()

    return train_loss
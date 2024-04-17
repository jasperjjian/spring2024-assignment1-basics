from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
import torch.nn as nn
import wandb

def train(model, train_loader, valid_loader, optimizer, criterion, steps, device):
    
    model.train()
    train_loss = 0.0
    #train_acc = 0.0

    for step, (inputs, targets) in enumerate(train_loader, start=1):
        if step > steps:
            break
        
        optimizer.zero_grad()
        preds = model(inputs)
        loss = criterion(preds, targets)
        loss.backward()
        train_loss = optimizer.step()

    return
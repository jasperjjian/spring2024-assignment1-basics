import torch
import torch.nn as nn
import numpy as np

class CrossEntropyLoss(nn.Module):
    def forward(self, inputs, target):
        flattened_inputs = inputs.view(-1, inputs.size(-1))
        batch_size = flattened_inputs.shape[0]
        flattened_targets = target.view(-1)
        
        # Normalize the logits
        maxed = flattened_inputs - torch.max(flattened_inputs, dim=-1, keepdim=True).values
        
        # Indexing the logits
        indexed_logits = maxed[torch.arange(maxed.size(0)), flattened_targets.long()]
        
        # Calculate the denominator
        exp_logits = torch.exp(maxed)
        denominator = torch.sum(exp_logits, dim=-1, keepdim=True)
        
        # Compute the loss
        loss = -indexed_logits + torch.log(denominator.squeeze())
        
        loss = torch.sum(loss) / batch_size
        
        return loss


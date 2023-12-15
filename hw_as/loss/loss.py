import torch
import torch.nn as nn

class WeightedCELoss(nn.Module):
    def __init__(self, weights = [9, 1], **kwargs):
        super().__init__()
        self.weights =torch.Tensor(weights)
    
    def forward(self, preds, labels, **kwargs):
        return nn.CrossEntropyLoss(weight=self.weights)(preds, labels)

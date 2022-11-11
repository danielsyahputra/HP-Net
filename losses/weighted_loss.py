import torch 
from torch import Tensor

class WeightedBCELoss(object):
    def __init__(self, weights = None) -> None:
        if type(weights).__name__ != "Tensor":
            self.weights = weights
        self.weights = weights

    def forward(self, outputs: Tensor, targets: Tensor, epoch):
        if self.weights is not None:
            current_weights = torch.exp(targets + (1 - targets * 2) * self.weights)
            loss = current_weights * (targets * torch.log(outputs))
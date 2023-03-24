import scipy
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps # If MSE == 0, We need eps

    def forward(self, yhat, y) -> Tensor:
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


def cross_entropy(y_pred: Tensor, y_true: Tensor) -> Tensor:
    criterion = nn.CrossEntropyLoss() # nn.CrossEntropyLoss: class
    return criterion(y_pred, y_true)


def binary_bce(y_pred: Tensor, y_true: Tensor) -> Tensor:
    criterion = nn.BCEWithLogitsLoss()
    return criterion(y_pred, y_true)


def pearson_loss(y_pred: np.ndarry, y_true: np.ndarry) -> Tensor:
    x = y_pred.clone()
    y = y_true.clone()
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    cov = torch.sum(vx * vy)
    corr = cov / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-12)
    corr = torch.maximum(torch.minimum(corr,torch.tensor(1)), torch.tensor(-1))
    return torch.sub(torch.tensor(1), corr ** 2)



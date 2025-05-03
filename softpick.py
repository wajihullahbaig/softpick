
import torch
import torch.nn as nn

# Forked from https://github.com/nkapila6/softpick/blob/main/softpick.py
class Softpick(nn.Module):
    def __init__(self, dim=-1, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
    
    def forward(self, x):
        mmax = torch.max(x, dim=self.dim, keepdim=True)
        m = mmax.values
        num = torch.exp(x-m) - torch.exp(-m)
        numer = torch.relu(num)
        d = torch.abs(num)
        denom = torch.sum(d, dim=self.dim, keepdim=True) + self.eps
        return numer/denom
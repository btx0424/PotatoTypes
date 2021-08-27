import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists

class Chain(nn.Module):
    def __init__(self, func_list: list) -> None:
        super().__init__()
        self.func_list = func_list
        self.module_list = nn.ModuleList([func for func in self.func_list if isinstance(func, nn.Module)])

    def forward(self, x):
        for func in self.func_list:
            if isinstance(x, tuple) or isinstance(x, list):
                x = func(*x)
            else:
                x = func(x)
        return x

    def __getitem__(self, i):
        if isinstance(i, slice):
            return Chain(self.func_list[i])
        else:
            return self.func_list[i]

class Split(nn.Module):
    def __init__(self, *branches) -> None:
        super().__init__()
        self.branches = branches
        self.module_list = nn.ModuleList([branch for branch in self.branches if isinstance(branch, nn.Module)])

    def forward(self, x):
        return tuple(branch(x) for branch in self.branches)

class Parallel(nn.Module):
    def __init__(self, *branches):
        super().__init__()
        self.branches = branches
        self.module_list = nn.ModuleList([branch for branch in self.branches if isinstance(branch, nn.Module)])
    
    def forward(self, *xs):
        assert len(xs)==len(self.branches)
        return tuple(f(x) for f, x in zip(self.branches, xs))
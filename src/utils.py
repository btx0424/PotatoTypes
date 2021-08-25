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
        return self.func_list[i]

class Parallel(nn.Module):
    def __init__(self, *branches) -> None:
        super().__init__()
        self.branches = branches
        self.module_list = nn.ModuleList([branch for branch in self.branches if isinstance(branch, nn.Module)])

    def forward(self, x):
        return [branch(x) for branch in self.branches]

class BayesianLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool=True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.logstd = nn.Parameter(torch.zeros_like(self.weight))

    def forward(self, x):
        mean = super().forward(x)
        std = F.linear(x**2, torch.exp(self.logstd))
        return dists.Normal(mean, std)

class BayesianConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, dilation=1, groups: int=1, bias: bool=True, padding_mode: str='zeros', device=None, dtype=None) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)
        self.logstd = nn.Parameter(torch.zeros_like(self.weight))
    
    def forward(self, x):
        mean = super().forward(x)
        std = F.conv2d(x**2, torch.exp(self.logstd), stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return dists.Normal(mean, std)

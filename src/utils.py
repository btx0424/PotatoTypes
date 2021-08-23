import torch.nn as nn

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

class Parallel(nn.Module):
    def __init__(self, *branches) -> None:
        super().__init__()
        self.branches = branches
        self.module_list = nn.ModuleList([branch for branch in self.branches if isinstance(branch, nn.Module)])

    def forward(self, x):
        return [branch(x) for branch in self.branches]
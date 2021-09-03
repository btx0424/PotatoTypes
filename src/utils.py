import torch
import torch
import torch.nn as nn
import torch.distributions as dists

def make_normal(mean, logstd):
    return dists.Normal(mean, torch.exp(logstd))

def chunk(n, dim):
    return lambda x: torch.chunk(x, n, dim)

def identity(x):
    return x

def cat(dim):
    return lambda *xs: torch.cat(xs, dim)

def reshape(*shape):
    return lambda x: torch.reshape(x, shape)

def sample(dist):
    return dist.rsample()

def output(to):
    def foo(x):
        to.append(x)
        return x
    return foo

def build_mlp(hidden_units: list, activation=nn.LeakyReLU) -> nn.Module:
    layers = []
    for in_features, out_features in zip(hidden_units, hidden_units[1:]):
        layers.append(nn.Linear(in_features, out_features))
        layers.append(activation())
    return nn.Sequential(*layers)
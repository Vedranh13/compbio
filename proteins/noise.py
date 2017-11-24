import torch

def add_gauss(mu=0, sig=1):
    def inner(ten):
        return ten + sig * torch.randn(ten.size()) + mu
    return inner


def add_uniform(l=-1, r=1):
    def inner(ten):
        s = r - l
        return ten + s * torch.rand(ten.size()) + l
    return inner

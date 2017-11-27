import torch
from torchvision.transforms import ToTensor, ToPILImage, Compose, Lambda
from PIL import Image

def add_gauss(mu=0, sig=1):
    def inner(ten):
        if isinstance(ten, Image.Image):
            ten = ToTensor().__call__(ten)
            ten = inner(ten)
            return ToPILImage().__call__(ten)
        return ten + sig * torch.randn(ten.size()) + mu
    return inner


def add_uniform(l=-1, r=1):
    def inner(ten):
        if isinstance(ten, Image.Image):
            ten = ToTensor().__call__(ten)
            ten = inner(ten)
            return ToPILImage().__call__(ten)
        s = r - l
        return ten + s * torch.rand(ten.size()) + l
    return inner


def add_gauss_spatial(mu=0, sig=1, cent=(50, 50)):
    pass


def make_noisy_tf(spread=1, noise='uniform'):
    if noise == 'uniform':
        noi = Lambda(add_uniform(l=-spread/2, r=-spread/2))
    elif noise == 'gauss' or 'gaussian':
        noi = Lambda(add_gauss(sig=spread))
    return Compose([ToTensor(), noi])

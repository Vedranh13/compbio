"""project_fst"""
from numpy.fft import fftn, ifftn, fftshift, ifftshift
from numpy import arange, linspace, meshgrid, array, vectorize, newaxis, moveaxis
from numpy import real, imag
import numpy.linalg as LA
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def project_fst(a, b, n, rho=None, rhoh=None):
    """"""
    a = a / LA.norm(a)
    b = b / LA.norm(b)
    if not rhoh:
        assert rho
        rhoh = fourier_lin(rho)
    data = sample_naive(rhoh, a, b, n=n)
    im = real(ifftn(ifftshift(data)))
    return im

def fourier_lin(ten):
    """Returns a function that returns the value of the DFT at a point.
    Uses piecewise linear interpolation"""
    tenh = fftn(ten)
    tenh = fftshift(tenh)
    reAl = real(tenh)
    fake = imag(tenh)
    n = ten.shape[0]
    x = arange(-n/2, n/2, 1) # why 128
    y = arange(-n/2, n/2, 1)
    z = arange(-n/2, n/2, 1)
    reg = RegularGridInterpolator((x, y, z), reAl, bounds_error=False)
    reg_fake = RegularGridInterpolator((x, y, z), fake, bounds_error=False)
    f = lambda a, b: complex(a, b)
    f = vectorize(f)
    true = lambda x: f(reg((x[0], x[1], x[2])), reg_fake((x[0], x[1], x[2])))
    return true


def sample_naive(func, a, b, n=50): # n should probs be smart
    x = arange(-n/2 + 1, n/2 - 1, 1) # * a
    y = arange(-n/2 + 1, n/2 - 1, 1) # * b
    z = linspace(0, 0, x.shape[0])
    f = lambda x: complex(x)
    f = vectorize(f)
    nx, ny = meshgrid(x, y)
    nx = nx.reshape((nx.shape[0], nx.shape[1], 1))
    nx = nx * a.T
    ny = ny.reshape((ny.shape[0], ny.shape[1], 1))
    ny = ny * b.T
    A = nx + ny
    A = moveaxis(A, -1, 0)
    data = func(A)
    data = f(data)
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot(xs=x, ys=y, zs = f(data))
    return data

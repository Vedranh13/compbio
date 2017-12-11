"""project_fst"""
from numpy.fft import fftn, ifftn, fftshift, ifftshift
from numpy import arange, linspace, meshgrid, array, vectorize, newaxis, moveaxis, abs
from numpy import real, imag
import numpy.linalg as LA
from scipy.interpolate import RegularGridInterpolator

def project_fst(a, b, n, rho=None, rhoh=None):
    """"""
    a = a / LA.norm(a)
    b = b / LA.norm(b)
    if not rhoh:
        assert rho is not None
        rhoh = fourier_lin(rho)
    data = sample_naive(rhoh, a, b, n=n)
    im = real(ifftn(ifftshift(data)))
    return im

def fourier_lin(ten):
    """Returns a function that returns the value of the DFT at a point.
    Uses piecewise linear interpolation"""
    tenh = fftn(ten)
    tenh = fftshift(tenh)
    n = ten.shape[0]
    if n % 2 == 0:
        a = arange(-(n)/2, n/2)
    else:
        a = arange(-(n - 1)/2, (n + 1)/2)
    x, y, z = meshgrid(a, a, a, indexing="ij")

    tenh *= (-1) ** abs(x + y + z)
    tenh /= n ** 3

    reAl = real(tenh)
    fake = imag(tenh)

    reg = RegularGridInterpolator((a, a, a), reAl, fill_value=0, bounds_error=False)
    reg_fake = RegularGridInterpolator((a, a, a), fake, fill_value=0, bounds_error=False)
    f = lambda a, b: complex(a, b)
    f = vectorize(f)
    true = lambda x: f(reg((x[0], x[1], x[2])), reg_fake((x[0], x[1], x[2])))
    return true


def sample_naive(func, a, b, n=50): # n should probs be smart
    if n % 2 == 0:
        x = arange(-(n)/2, n/2 )
        y = arange(-(n)/2, n/2)
    else:
        x = arange(-(n - 1)/2 , (n + 1)/2, 1) # * a
        y = arange(-(n - 1)/2 , (n + 1)/2, 1) # * b
    # z = linspace(0, 0, x.shape[0])
    f = lambda x: complex(x)
    f = vectorize(f)
    nx, ny = meshgrid(x, y, indexing='ij')
    # nx = nx.reshape((nx.shape[0], nx.shape[1], 1))
    nxp = nx[..., newaxis]
    nxp = nxp * a
    # ny = ny.reshape((ny.shape[0], ny.shape[1], 1))
    nyp = ny[..., newaxis]
    nyp = nyp * b
    A = nxp + nyp
    A = moveaxis(A, -1, 0)
    data = func(A)
    data = f(data)
    data *= (-1) ** abs(nx + ny)
    data /= n ** 3
    return data

"""reconstruct"""
import numpy as np
from numpy.fft import fftn, ifftn, fftshift, ifftshift
from numpy import arange, linspace, meshgrid, array, vectorize, newaxis, moveaxis, abs
from numpy import real, imag
import numpy.linalg as LA
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import gaussian
from scipy.misc import imread
from utils import gen_dataset, gen_randim_and_axis, read_dict, save_dict, load_protein
import os
import sys
from project_FST import fourier_lin, project_fst
from numpy import einsum as ein
import matplotlib.pyplot as plt

from math import sqrt

import time

t = lambda: time.time()
"""
planning to allow 'gaussian' and 'uniform' as inputs to the 'method' parameter

"""
def smear_proj(img, L, method='gaussian',sigma=None):
    methods = ['gaussian', 'uniform']
    if method not in methods:
        method = 'gaussian'

    window = None
    if method == 'gaussian':
        if not sigma:
            sigma = L/10
        window = gaussian(L, sigma)
    elif method == 'uniform':
        window = np.ones(L)/L

    #c, a,b coords, i.e. the vertical axis comes first. weird, but make visualizing the array easier
    return ein('jk,i', img, window) #weird notation, but works pretty well. This does the smearing.   

"""
We are given a 3-d image with basis vectors a and b as the direction along the second and third dimensions.
The first dimension is 'vertical' i.e. the cross of those two (this makes visualizing the matrix easier)

This function expects a, b, and c to be roughly orthonormal
"""
def smear_to_common_coords(smeared, av, bv, cv):
    n = smeared.shape[0]
    if n % 2 == 0:
        a = arange(-(n)/2, n/2)
    else:
        a = arange(-(n - 1)/2, (n + 1)/2)
    
    reg = RegularGridInterpolator((a, a, a), smeared, fill_value=0, bounds_error=False)
    #a, b, and c are orthonormal, so we can dot them with any vector "v" describing a point in the standard basis 
    #to get the corresponding point in a-b-c space.

    count = -64
    c_time = t()
    mat = np.stack([cv,av,bv])
    #as stated above, coords are in x-y-z space. This function converts the coords to a-b-c space
    #and returns the value from the smeared image by using the grid interpolator.
    def f(*coords):
        x = np.array(coords)
        nonlocal count, c_time
        #x = coords
        temp_v = coords[1]
        if temp_v != count:
            count = temp_v
            temp_t = t()
            diff = temp_t - c_time
            c_time = temp_t
            print('iteration took', diff, 'seconds, curr iter:',temp_v)
        #print(x.shape)
        #print(a.shape)
        #print(b.shape)
        #print(c.shape)
        return reg(np.dot(mat,x))
    f = np.vectorize(f)
    #arr = np.zeros(n,n,n)
    xv, yv, zv = np.meshgrid(a,a,a, indexing='ij')
    #for i in len(a):
    #    for j in len(a):
    #        for k in len(a):
    #            arr[i,j,k]
    return f(zv, xv, yv)

def mydot(tupa, tupb):
    return tupa[0] * tupb[0] + tupa[1]*tupb[1] + tupa[2] * tupb[2]
"""
temporary test version of reconstruct: all code in here should be fully implemented
"""
def reconstruct_t(imgs, orientations):
    img, ori = imgs[0], orientations[0]
    print('starting smear')
    smeared = smear_proj(img, img.shape[0])
    print('finished smear')
    a = ori[0]
    b = ori[0]

    a = a/sqrt(np.dot(a,a))
    b = b/sqrt(np.dot(b,b))
 
    c = np.cross(a,b)
    #c = c/sqrt(np.dot(c,c)) #shouldn't make a difference... but something something double precision errors
    print('converting to common coords...')
    smeared = smear_to_common_coords(smeared, a, b, c)
    print('finished. Projecting...')
    in_common_coords = project_fst(np.array([1,0,0]), np.array([0,1,0]), img.shape[0], rho=smeared)
    print('done. Saving...')
    dct = {'smeared_test' : in_common_coords}
    save_dict(dct, dir='test_reconstruction')
    

"""
first parameter is images, second is list of corresponding a,b vectors for orientation

"""
def reconstruct(imgs, orientations=None):

    #we need orientations, whether given or not. If not given, reconstruct them with the common line technique
    #I also think we calculate the fourier transforms of l_j in common line? if not, can take that bit out
    need_lj = True
    lj_fourier = []
    if not orientations:
        lj_fourier, orientations = common_line(imgs)
        need_lj = False
    #step 1
    #print('RGI')
    assert len(imgs) > 0
    L = imgs[0].shape[0]
    sum_smears = np.zeros(L,L,L)

    for img, pair in zip(img_orient_bundle, orientations):
        a = pair[0]
        b = pair[1]
        a = a/sqrt(np.dot(a,a))
        b = b/sqrt(np.dot(b,b))
 
        c = np.cross(a,b)
        c = c/sqrt(np.dot(c,c)) #shouldn't make a difference... but something something double precision errors

        #step 2
        smeared = smear_proj(img,L, method='gaussian')

        #step 3
        sum_smears += smear_to_common_coords(smeared, a, b, c)

        if need_lj:
            #start step 4/4a. see 4a for implementation details of get_lj_fourier
            lj_fourier.append(get_lj_fourier(c, L))

    #complete step 4
    sum_smears_fourier = fourier_lin(sum_smears)
    sum_lj_fourier = sum(lj_fourier, np.zeros(lj_fourier[0].shape))
    
    return compute_final_rho(sum_smears_fourier, sum_lj_fourier)


def create_dataset(dir):
    gen_dataset(proteins=["zika", "mystery"], n_each=10, write=True, dir=dir)


def test_reconstruction():
    print('not fully implemented')
    

def plot_image(filename):
    arr = imread(filename, mode='L')
    print(arr.shape)
    plt.imshow(arr, cmap='gray')
    plt.show()

if __name__ == '__main__':
    #print(os.listdir('dataset'))
    d = 'dataset'
    stuff = os.listdir(d)
    if len(stuff) == 0:
        create_dataset(d)
    print(stuff[3])
    #arr = imread(d + '/' + stuff[3], mode='L')
    #print(arr.shape)
    #plt.imshow(arr, cmap='gray')
    #plt.show()
    if len(sys.argv) > 1 and sys.argv[1] == 'p':
        plot_image('test_reconstruction/smeared_test.png')
    else:
        rho = load_protein('mystery.mrc')
        a, b, image = gen_randim_and_axis(rho, rho.shape[0])
        print('starting reconstruction')

        #put in the format expected by reconstruct_t
        imgs = [image]
        ori = [(a,b)]
        reconstruct_t(imgs, ori)
        print('Done')
    
"""

Basic idea in a few steps:

1. start off with list of projections (p), and a list of corresponding orientation (a,b) tuples. Note that later on we will have a more
 general reconstruction method that just takes in projections.
2. create grid of values in (a, b, a x b) space where the a-b plane is the center of a bunch of 1-d guassians that integrate to whatever 
        the projection value was at that point. Basically set the image flat on a plane and gaussian-smear it up and down.
3. use regular grid interpolator to move from (a,b, a x b) to (x,y,z).
4. We aim to use the eqn: 3d obj  =  inv Fourier (  Fourier ( sum of smears) / Fourier (sum of l_j))
 4a. We directly calculate  Fourier (sum of smears) , but for Fourier (sum of l_j)  we use:
        F{l_j} = L sinc(L * pi * w_x^j)    Note that F{l_j} is a function of w_x^j, and that it computes a one-dimensional vector. still
                   trying to figure out how this translates to 3d space and how to "divide" the results (element wise? matrix inversion?
5. use project_fst, feeding in the initial directions to create comparison images. do a visual check from a bunch of different angles.
"""

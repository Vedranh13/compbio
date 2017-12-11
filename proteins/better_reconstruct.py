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
import mrcfile

from math import sqrt

import time

t = lambda: time.time()

def common_line(imgs):
    ori = []
    N = imgs[0].shape[0]
    #define first image to lie in the y-z plane
    ori.append((np.array([0,1,0]), np.array([0,0,1])))

    if len(imgs) == 1:
        return ori

    #for at least two images, we let the common line between them be the z-axis
    l12 = np.array([0,0,1])

    #gets us a list of functions that can be queried for fourier transform points
    fourier_imgs = [fourier_lin_2d(img)[0] for img in imgs]
    print('finished fourier transforms')
    img1 = fourier_imgs[0]
    img2 = fourier_imgs[1]
    
    l12_theta, l21_theta = get_local_angles(img1, img2, N)
    print('got first angles')
    if len(imgs) == 2:
        #we can't actually reconstruct with two images, but we can give at least a consistent guess
        ori.append((np.array([1,0,0]), np.array([0, np.cos(l12_theta), np.sin(l12_theta)])))
        return ori
    #now we have the two best angles in terms of the original images
    
    img_3 = fourier_imgs[2]
    l23_theta, l32_theta = get_local_angles(img2, img_3, N)
    l13_theta, l31_theta = get_local_angles(img1, img_3, N)
    print('finished second batch of angles')
    #using Herman's algorithm
    A = abs(l32_theta - l31_theta)
    B = abs(l21_theta - l23_theta)
    C = abs(l12_theta - l13_theta)

    if A > np.pi/2:
        A = np.pi - A
    if B > np.pi/2:
        B = np.pi - B
    if C > np.pi/2:
            C = np.pi - C

    if A == 0:
        A = 0.001
    if B == 0:
        B = 0.001
    if C == 0:
        C = 0.001
    #angle between images 1 and 2. 
    alpha = np.arccos((np.cos(A) - np.cos(B)*np.cos(C))/(np.sin(B)*np.sin(C)))
        
    #angle between 1 and 3
    beta = np.arccos((np.cos(B) - np.cos(A)*np.cos(C))/(np.sin(A) * np.sin(C)))

    #angle between 2 and 3
    gamma = np.arccos((np.cos(C) - np.cos(A)*np.cos(B))/(np.sin(A) * np.sin(B)))
        
    #first vector encodes the angle between image 2 and the y-z plane (i.e. image 1), 
    #second vector encodes this image's common line in the yz plane which is why we use the local angle from image 1.
    ori.append((np.array([-1 * np.sin(alpha), np.cos(alpha), 0]), np.array([0, np.cos(l12_theta), np.sin(l12_theta)])))

    V13 = np.array([0, np.cos(l13_theta), np.sin(l13_theta)])
    V23 = np.array([0, np.cos(l23_theta), np.sin(l23_theta)])
    V_mat = np.stack([V13, V23]).T

    l_31 = np.array([np.cos(l31_theta), np.sin(l31_theta)])
    l_32 = np.array([np.cos(l32_theta), np.sin(l32_theta)])
    l_mat_inv = np.stack([l_31, l_32])

    new_vecs_mat = np.dot(V_mat, l_mat_inv)
    v1 = new_vecs_mat[:,0]
    v2 = new_vecs_mat[:,1]
    v1 = v1 /LA.norm(v1)
    v2 = v2 /LA.norm(v2)
    ori.append((v1, v2))
    
    for i in range(3, len(imgs)):
        img_i = fourier_imgs[i]
        
        l1i_theta, li1_theta = get_local_angles(img1, img_i, N)
        l2i_theta, li2_theta = get_local_angles(img2, img_i, N)

        V1i = np.array([0, np.cos(l1i_theta), np.sin(l1i_theta)])
        V2i = np.array([0, np.cos(l2i_theta), np.sin(l2i_theta)])
        V_mat = np.stack([V1i, V2i]).T

        l_i1 = np.array([np.cos(li1_theta), np.sin(li1_theta)])
        l_i2 = np.array([np.cos(li2_theta), np.sin(li2_theta)])
        l_mat_inv = np.stack([l_i1, l_i2])

        new_vecs_mat = np.dot(V_mat, l_mat_inv)
        v1 = new_vecs_mat[:,0]
        v2 = new_vecs_mat[:,1]
        v1 = v1 /LA.norm(v1)
        v2 = v2 /LA.norm(v2)
        ori.append((v1, v2))
    return ori

def get_local_angles(img1, img2, N): 
    best_pair = (-1,-1)
    best_dot = -1 # want a high magnitude
    for theta1 in np.arange(0,np.pi, np.pi/10):
        line1 = get_line(img1, theta1, N)
        for theta2 in np.arange(0, np.pi, np.pi/10):
            line2 = get_line(img2, theta2, N)
            temp = np.absolute(np.vdot(line1, line2)/(LA.norm(line1) * LA.norm(line2)))
            if temp > best_dot:
                temp = best_dot
                best_pair = (theta1, theta2)
    return best_pair

#img is a function (back-end is an RGI) that lets us query an image's fourier transform
#this function gets a slice of that image
def get_line(img, theta, n):
    vec = np.zeros(n, dtype=np.complex128)
    norm_dir = np.array([np.cos(theta), np.sin(theta)])
    if n % 2 == 0:
        a = arange(-(n)/2, n/2)
    else:
        a = arange(-(n - 1)/2, (n + 1)/2)
    
    for i in range(n):
        vec[i] = img(a[i] * norm_dir)
    return vec

def fourier_lin_2d(ten):
    """Returns a function that returns the value of the DFT at a point.
    Uses piecewise linear interpolation"""
    tenh = fftn(ten)
    tenh = fftshift(tenh)
    n = ten.shape[0]
    if n % 2 == 0:
        a = arange(-(n)/2, n/2)
    else:
        a = arange(-(n - 1)/2, (n + 1)/2)
    x, y = meshgrid(a, a, indexing="ij")

    tenh *= (-1) ** abs(x + y)
    tenh /= n ** 2

    reAl = real(tenh)
    fake = imag(tenh)

    reg = RegularGridInterpolator((a, a), reAl, fill_value=0, bounds_error=False)
    reg_fake = RegularGridInterpolator((a, a), fake, fill_value=0, bounds_error=False)
    f = lambda a, b: complex(a, b)
    f = vectorize(f)
    true = lambda x: f(reg((x[0], x[1])), reg_fake((x[0], x[1])))
    return true, a

def smear_fourier(img, N, sinc_mat):
    im_fourier, interval = fourier_lin_2d(img)
    im_fourier_calc = np.zeros((N,N), dtype='complex')
    for i in range(N):
        for j in range(N):
            im_fourier_calc[i,j]= im_fourier((i,j))

    
    #speed this up with numpy?
    im_fourier_stacked = np.stack([im_fourier_calc for i in range(N)], axis=-1)
    
    #print(sinc_mat.shape)
    #print(im_fourier_stacked.shape)
    return np.multiply(im_fourier_stacked,sinc_mat), sinc_mat, interval

#a, b, and c are orthonormal
"""
    real_part = real(fourier_smear)
    imag_part = imag(fourier_smear)
    reg = RegularGridInterpolator((axis, axis, axis), real_part, fill_value=0, bounds_error=False)
    reg_imag = RegularGridInterpolator((axis, axis, axis), imag_part, fill_value=0, bounds_error=False)

    xv, yv, zv = np.meshgrid(axis, axis, axis, indexing='ij')
    coordmat = np.stack((xv,yv,zv), axis=-1)
    basismat = np.stack((a,b,c)).T # need column vectors
    
    new_coords = np.dot(coordmat, basismat)
    c_time = t()
    vals = np.zeros((n,n,n), dtype='complex')
    for x in range(n):
        for y in range(n):
            for z in range(n):
                vals[x,y,z] = complex(reg(new_coords[x,y,z,:]), reg_imag(new_coords[x,y,z,:]))
        temp_t = t()
        diff = temp_t - c_time
        c_time = temp_t
        print('iteration took', diff, 'seconds, curr iter:', x)
    return vals
    """
def to_standard_basis(fourier_smear, a, b, c, n, axis):
    reg = RegularGridInterpolator((axis, axis, axis), fourier_smear, fill_value=0, bounds_error=False)
    #a, b, and c are orthonormal, so we can dot them with any vector "v" describing a point in the standard basis
    #to get the corresponding point in a-b-c space.

    count = -64
    c_time = t()
    mat = np.stack((a,b,c))
    #as stated above, coords are in x-y-z space. This function converts the coords to a-b-c space
    #and returns the value from the smeared image by using the grid interpolator.

    my_reg = make_nearest_interp(fourier_smear, n)
    def f(*coords):
        x = np.array(coords)

        ##Just for timing purposes ##
        #nonlocal count, c_time
        #temp_v = coords[0]
        #if temp_v != count:
        #    count = temp_v
        #    temp_t = t()
        #    diff = temp_t - c_time
        #    c_time = temp_t
        #    print('iteration took', diff, 'seconds, curr iter:',temp_v)
        ################################################this does all the work
        #return reg(np.dot(mat,x))
        #return reg(np.dot(mat,x), method='nearest')
        return my_reg(np.dot(mat,x))
    f = np.vectorize(f)
    xv, yv, zv = np.meshgrid(axis,axis,axis, indexing='ij')
    return f(xv, yv, zv)

def make_nearest_interp(mat, N):
    zero = complex(0,0) if mat.dtype == np.complex128 else 0
    def interp(coords):
        coord0 = int(round((coords[0]) + N/2))
        coord1 = int(round((coords[1]) + N/2))
        coord2 = int(round((coords[2]) + N/2))
        if coord0 >= N or coord1 >= N or coord2 >= N or coord0 < 0 or coord1 < 0 or coord2 < 0:
            return zero
        return mat[coord0, coord1, coord2]
    return interp

"""
temporary test version of reconstruct: all code in here should be fully implemented
"""
def reconstruct_t(imgs, orientations):
    img, ori = imgs[0], orientations[0]
    N = img.shape[0]

    if N % 2 == 0:
        interval = arange(-(N)/2, N/2)
    else:
        interval = arange(-(N - 1)/2, (N + 1)/2)

    #calculate fourier transform of l_j
    sinc_interval = (interval/interval[-1])*np.pi #frequency range for sinc is -pi to pi
    sinc_interval = sinc_interval *N * np.pi #in the formula, the argument get muliplied by N pi 
                                             #if this doesn't work, try just multiplying by N.
    sinc_vec = N * np.sinc(sinc_interval)
    sinc_mat = np.stack([np.ones((N,N)) * val for val in sinc_vec], axis=-1)
    print('starting smear')
    fourier_smeared, sinc_mat, interval = smear_fourier(img, N,sinc_mat)
    print('finished smear')
    a = ori[0]
    b = ori[0]

    a = a/sqrt(np.dot(a,a))
    b = b/sqrt(np.dot(b,b))
 
    c = np.cross(a,b)
    #c = c/sqrt(np.dot(c,c)) #shouldn't make a difference... but something something double precision errors
    print('converting to common coords...')
    smeared = to_standard_basis(fourier_smeared, a, b, c, N, interval)
    print('finished. Projecting...')
    #in_common_coords = project_fst(np.array([1,0,0]), np.array([0,1,0]), img.shape[0], rho=smeared)
    in_common_coords = real(ifftn(ifftshift(smeared)))
    im_test = project_fst(a, b, N, rho=in_common_coords)
    in_common_coords = in_common_coords.astype(np.float32)
    print('done. Saving...')
    dct = {'smeared_test' : im_test}
    save_dict(dct, dir='test_reconstruction')
    with mrcfile.new('tmp.mrc', overwrite=True ) as mrc:
        mrc.set_data(in_common_coords)
    

"""
first parameter is images, second is list of corresponding a,b vectors for orientation
"""
def reconstruct(imgs, orientations=None):
    N = imgs[0].shape[0]

    if orientations is None:
        #common line technique, but for now just cry
        raise RuntimeException('must provide orientations')

    if N % 2 == 0:
        interval = arange(-(N)/2, N/2)
    else:
        interval = arange(-(N - 1)/2, (N + 1)/2)

    #calculate fourier transform of l_j
    sinc_interval = (interval/interval[-1])*np.pi #frequency range for sinc
    sinc_vec = np.sinc(sinc_interval)
    sinc_mat = np.stack([np.ones((N,N)) * val for val in sinc_vec], axis=-1)

    b_j_sum = np.zeros((N,N,N), dtype=np.complex128)
    l_j_sum = np.zeros((N,N,N))
    count = 1
    curr_t = t()
    for img, pair in zip(imgs, orientations):
        fourier_smeared, _, _ = smear_fourier(img, N, sinc_mat)
        a = pair[0]
        b = pair[1]
        a = a/np.linalg.norm(a)
        b = b/np.linalg.norm(b)
 
        c = np.cross(a,b)
        
        b_j_sum += to_standard_basis(fourier_smeared, a, b, c, N, interval)
        print('halfway there')
        #b_j_sum += fourier_smeared
        #l_j_sum += sinc_mat
        l_j_sum += to_standard_basis(sinc_mat, a, b, c, N, interval)
        print(count)
        count += 1
    fin_t = t()
    print('execution took', fin_t - curr_t, 'seconds')
    l_j_sum[np.where(l_j_sum == 0)] = 1 #unfortunately the mrc isn't viewable unless I do this
    x, y, z = meshgrid(interval, interval, interval, indexing="ij")
    quot = b_j_sum
    quot = np.divide(b_j_sum, l_j_sum)
    quot[np.where(quot == np.inf)]= 0
    quot[np.where(quot == np.nan)] = 0
    quot = np.nan_to_num(quot)
    #next try removing z vv
    quot *= (-1) ** abs(x+y+z) 
    quot /= N**3
    
    return real(ifftn(ifftshift(quot))) # should always be real, but we want the datatype to switch too

def create_dataset(dir):
    gen_dataset(proteins=["zika", "mystery"], n_each=20, write=True, dir=dir)

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
    elif len(sys.argv) > 1 and sys.argv[1] == 'c':
        rho = load_protein('zika.mrc')
        #a, b, image = gen_randim_and_axis(rho, rho.shape[0])
        imgs = []
        ori = []
        for i in range(15):
            a, b, image = gen_randim_and_axis(rho, rho.shape[0])
            ori.append((a,b))
            imgs.append(image)
        ori2 = common_line(imgs)
        #print('these were the recovered orientations')
        #print(ori2)
        #print([(LA.norm(v[0]), LA.norm(v[1])) for v in ori2])
        #print('these were the actual orientations')
        #print(ori)
        #print('note that it is very likely to be off by a univeral rotation')
        final_img = reconstruct(imgs, ori2)
        in_common_coords = final_img.astype(np.float32)
        print('done with common line version. Saving...')

        with mrcfile.new('final_commonline.mrc', overwrite=True ) as mrc:
            mrc.set_data(in_common_coords)
        
        print('beginning normal reconstruction')
        final_img = reconstruct(imgs, ori)
        in_common_coords = final_img.astype(np.float32)
        print('done. Saving...')

        with mrcfile.new('final_nocommonline.mrc', overwrite=True ) as mrc:
            mrc.set_data(in_common_coords)
        
        print('Done')
        
        
    else:
        rho = load_protein('zika.mrc')
        #a, b, image = gen_randim_and_axis(rho, rho.shape[0])
        imgs = []
        ori = []
        for i in range(20):
            a, b, image = gen_randim_and_axis(rho, rho.shape[0])
            ori.append((a,b))
            imgs.append(image)
            
        print('starting reconstruction')

        #put in the format expected by reconstruct_t
        #imgs = [image]
        #ori = [(a,b)]
        final_img = reconstruct(imgs, ori)
        in_common_coords = final_img.astype(np.float32)
        print('done. Saving...')

        with mrcfile.new('final.mrc', overwrite=True ) as mrc:
            mrc.set_data(in_common_coords)
        
        print('Done')
    

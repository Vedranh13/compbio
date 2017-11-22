"""Various util functions"""
import numpy as np
import mrcfile
import matplotlib.pyplot as plt
from project_FST import fourier_lin, project_fst

def generate_protein_gauss(n):
    """This simulates a protein's 3D energy potential as just a gaussian blob"""
    return 5 * np.rand.randn(n, n, n)


def load_protein(filename):
    zika_file = mrcfile.open(filename)
    rho = zika_file.data
    return rho

# Mystery is a protein I used to know, og name was EMD-2830.map

def gen_dataset(proteins=["zika", "mystery"], n_each=1000, write=True):
    ims = {}
    for protein in proteins:
        rho = load_protein(protein + ".mrc")
        rhoh = fourier_lin(rho)
        for i in range(n_each):
            ims[protein + "_" + str(i)] = gen_randim(rhoh, rho.shape[0])
    if write:
        save_dict(ims)
    return ims

def gen_randim(rhoh, n):
    a_comp = [np.random.randint(0, 3)]
    b_comp = [i for i in range(3) if i not in a_comp]
    a = np.random.rand(3)
    b = np.random.rand(3)
    a[a_comp] = 0
    b[b_comp] = 0
    return project_fst(a, b, n, rhoh=rhoh)


def save_dict(dct, dir='dataset'):
    for k,v in dct.items():
        plt.imsave(dir + '/' + k + ".png", v, format="png", cmap=plt.cm.gray)

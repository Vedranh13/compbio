"""Various util functions"""
import numpy as np
import mrcfile

def generate_protein_gauss(n):
    """This simulates a protein's 3D energy potential as just a gaussian blob"""
    return 5 * np.rand.randn(n, n, n)


def load_protein(filename):
    zika_file = mrcfile.open(filename)
    rho = zika_file.data
    return rho

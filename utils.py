import numpy as np
from distance import hamming as ham
from Bio import Phylo
from io import StringIO   #if you are writing in python 2, use this: from cStringIO import StringIO
import  matplotlib
import matplotlib.pyplot as plt

def draw(treedata, output_file=""):
    handle = StringIO(treedata)
    tree = Phylo.read(handle, 'newick')
    matplotlib.rc('font', size=6)
    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    Phylo.draw(tree, axes=axes, do_show=False)
    plt.show()
    #plt.savefig(output_file+'.png')
    #plt.savefig(output_file+’.pdf’, format=‘PDF’)    #if you want to save as pdf file


def get_comp(base_pair):
    assert base_pair in ("A", "G", "T", "C")
    if base_pair == "A":
        return "G"
    if base_pair == "T":
        return "C"
    if base_pair == "C":
        return "T"
    if base_pair == "G":
        return "A"


def get_rand_ortho(base_pair, bias = .5):
    r = np.random.uniform()
    if base_pair == "A" or base_pair == "G":
        if r < bias:
            return "C"
        else:
            return "T"
    else:
        if r < bias:
            return "A"
        else:
            return "G"


def jc_distance(str1, str2):
    p = float(ham(str1, str2)) / float(len(str1))
    return -3.0 / 4.0 * np.log(1 - (4.0 / 3.0) * p)


def gen_base_strand(length):
    """Returns a 1/4 1/4 1/4 1/4 strand"""
    bases = []
    for _ in range(length):
        r = np.random.uniform()
        if r < .25:
            bases.append("A")
        elif r < .5:
            bases.append("G")
        elif r < .75:
            bases.append("T")
        else:
            bases.append("C")
    return "".join(bases)


def newick_cherry(f, g, df, dg):
    return "(" + str(f) + ":" + str(df) + "," + str(g) + ":" + str(dg) + ")"

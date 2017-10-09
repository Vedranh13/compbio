import numpy as np


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

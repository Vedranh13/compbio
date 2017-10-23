"""Here we create the data"""
import numpy as np
from utils import get_comp
from utils import get_rand_ortho
from scipy.stats import gennorm


def mutate_site(bp, com, intra = .75):
    trans = (1 - intra)
    r = np.random.uniform()
    if r < com:
        r = np.random.uniform()
        if r < intra:
            return get_comp(bp)
        else:
            return get_rand_ortho(bp)
    return bp


def mutate_site_jc(bp, mu):
    r = np.random.uniform()
    pairs = ["A", "G", "T", "C"]
    pairs.remove(bp)
    if r < mu:
        return pairs[0]
    if r < 2*mu:
        return pairs[1]
    if r < 3*mu:
        return pairs[2]
    return bp


def mutate_str_uniform(seq, com, intra = .75):
    return "".join([mutate_site(bp, com, intra) for bp in seq])


def mutate_site_ik(seq, i, k, com, intra = .75):
    return "".join([ mutate_site(seq[c], com, intra) if c >= i and c < k else seq[c] for c in range(len(seq))])


def mutate_site_ik_gauss(seq, i, k, com, intra=.75):
    return "".join([mutate_site(seq[i], gennorm.pdf(c, 2, (i + k) / 2, 1.0 / com)) for c in range(len(seq))])


def mutate_site_ik_laplace(seq, i, k, com, intra=.75):
    return "".join([mutate_site(seq[i], gennorm.pdf(c, 1, (i + k) / 2, 1.0 / com)) for c in range(len(seq))])


def mutate_str(seq, mu):
    return "".join([mutate_site_jc(bp, mu) for bp in seq])


def create_k_copies(lst, k):
    """Creates k copies of each of the n strands"""
    nk_lst = []
    for seq in lst:
        for _ in range(k):
            nk_lst.append(seq)
    return nk_lst


def mutate_samples_uniform(lst, k, com):
    nk = create_k_copies(lst, k)
    data = []
    for seq in nk:
        data.append(mutate_str(seq, com))
    return data

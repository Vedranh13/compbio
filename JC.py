""""Jukes Cantor simulator"""
import numpy as np
from sim_data import mutate_str

def pr_mutation(epoch, mu):
    v = epoch * mu
    e = np.exp(-1 * v)
    return 1/4 - 1/4 * e


def evolve_to_t(strand, t, mu):
    p = pr_mutation(t, mu)
    return mutate_str(strand, p)


def split_to_epoch(strand, epoch, mu):
    return evolve_to_t(strand, epoch, mu), evolve_to_t(strand, epoch, mu)


def evolve(base_strand, tree_of_splits, mu):
    if len(tree_of_splits) < 3:
        return [base_strand]
    left, right = split_to_epoch(base_strand, tree_of_splits[0], mu)
    left = evolve(left, tree_of_splits[1], mu)
    right = evolve(right, tree_of_splits[2], mu)
    return left + right

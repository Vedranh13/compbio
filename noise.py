import sim_data as sd
import NJ
from sklearn.cluster import DBSCAN as db
from Bio import Phylo
from io import StringIO
import numpy as np
import JC
import utils
def join_noised(n_seq, k, com=.05):
    data = sd.mutate_samples_uniform(n_seq, k, com)
    return NJ.join(data)


"""Now, we take that tree and try to learn the original species by clustering the tree"""
def create_metric(tree):
    def metric(x, y):
        i, j = int(x[0]), int(y[0])
        return tree.distance(i, j)
    return metric


def cluster(tree_newick):
    tree = Phylo.read(StringIO(tree_newick), 'newick')
    N = tree.count_terminals()
    x = np.arange(1, N + 1).reshape(-1, 1)
    for eps in np.linspace(0.1, tree.total_branch_length(), num=100):
        clustered = db(eps=eps, metric=create_metric(tree)).fit(x)


np.random.seed(123451234)
cs61a = [100, [50, [], []], [100, [], []]]
strands = JC.evolve(utils.gen_base_strand(30), cs61a, .0005)
cluster(join_noised(strands, 10, com=.005))

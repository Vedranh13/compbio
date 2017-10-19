
# coding: utf-8

# In[7]:


import numpy as np
from utils import jc_distance
from utils import newick_cherry
# DTABLE = np.array([[0, 5, 9, 9, 8], [5, 0, 10, 10, 9], [9, 10, 0, 8, 7], [9, 10, 8, 0, 3], [8, 9, 7, 3, 0]])
DTABLE = np.array([[0, .31, 1.01, .75, 1.03], [.31, 0, 1, .69, .9], [1.01, 1, 0, .61, .42], [.75, .69, .61, 0, .37], [1.03, .9, .42, .37, 0]])


# In[8]:


DTABLE


# In[9]:


def row_sums(DTABLE):
    Rs = [row.sum() for row in DTABLE]
    return Rs
def calc_Q(DTABLE):
    n = DTABLE.shape[0]
    Rs = row_sums(DTABLE)
    Q = []
    for i in range(DTABLE.shape[0]):
        for j in range(DTABLE.shape[1]):
            Q.append((n - 2)*DTABLE[i][j] - Rs[i] - Rs[j])
    Q = np.array(Q).reshape((n, n))
    return Q

def cherry(DTABLE, f, g):
    n = DTABLE.shape[0]
    Rs = row_sums(DTABLE)
    NEW_D = []
    DFU = (1/2)*DTABLE[f][g] + (1/(2*(n - 2))) * (Rs[f] - Rs[g])
    DGU = DTABLE[f][g] - DFU
    for i in range(DTABLE.shape[0]):
        for j in range(DTABLE.shape[1]):
            if (i == g) or (j == g):
                continue
            elif i == f:
                NEW_D.append((1/2)*(DTABLE[f][j] + DTABLE[g][j] - DTABLE[f][g]))
            elif j == f:
                NEW_D.append((1/2)*(DTABLE[f][i] + DTABLE[g][i] - DTABLE[f][g]))
            else:
                NEW_D.append(DTABLE[i][j])
    NEW_D = np.array(NEW_D).reshape((DTABLE.shape[0] - 1, DTABLE.shape[0] - 1))
    return NEW_D, DFU, DGU

def find_min(X):
    min = 99999999
    minIJ = None
    for i in range(len(X)):
        for j in range(len(X[0])):
            if X[i][j] < min and i != j:
                min = X[i][j]
                minIJ = (i, j)
    return minIJ

def get_all_cherries(DTABLE):
    n = DTABLE.shape[0]
    for i in range(n - 2):
        Q = calc_Q(DTABLE)
        f,g = find_min(Q)
        res = cherry(DTABLE, f, g)
        yield (f, g, res[1:])
        DTABLE = res[0]

def get_DTABLE(lst_strands):
    DTABLE = []
    for i in lst_strands:
        row = []
        for j in lst_strands:
            row.append(jc_distance(i, j))
        DTABLE.append(row)
    return np.matrix(DTABLE)

def join(lst_strands):
    DTABLE = get_DTABLE(lst_strands)
    N = len(lst_strands)
    in_to_real = {a : a for a in range(1, N + 1)}
    for f, g, DFU, DGU in get_all_cherries(DTABLE):
        f += 1
        g += 1
        spec1 = in_to_real[f]
        spec2 = in_to_real[g]
        for spec in range(g + 1, N):
            in_to_real[spec] = in_to_real[spec + 1]
        in_to_real[f] = newick_cherry(spec1, spec2, DFU, DGU)
    return in_to_real[1]

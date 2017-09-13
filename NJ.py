
# coding: utf-8

# In[61]:


import numpy as np
DTABLE = np.array([[0, .31, 1.01, .75, 1.03], [.31, 0, 1, .69, .9], [1.01, 1, 0, .61, .42], [.75, .69, .61, 0, .37], [1.03, .9, .42, .37, 0]])


# In[68]:


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
    return calc_Q(NEW_D), DFU, DGU


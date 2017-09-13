
# coding: utf-8

# In[28]:


import numpy as np
DTABLE = np.array([[0, .31, 1.01, .75, 1.03], [.31, 0, 1, .69, .9], [1.01, 1, 0, .61, .42], [.75, .69, .61, 0, .37], [1.03, .9, .42, .37, 0]])


# In[32]:


Rs = [row.sum() for row in DTABLE]


# In[39]:


n = DTABLE.shape[0]
Q = []
for i in range(DTABLE.shape[0]):
    for j in range(DTABLE.shape[1]):
        Q.append((n - 2)*DTABLE[i][j] - Rs[i] - Rs[j])
Q = np.array(Q).reshape((n, n))
Q


from scipy.io import loadmat
from NJ import join
from utils import draw
data = loadmat('flhivdata.mat')
lst_strands = []
minLen = 9999999999
int_to_name = {}
i = 1
for key, value in data.items():
    if "__" in key:
        continue
    value = str(value[0])
    if (len(value) < minLen):
        minLen = len(value)
    lst_strands.append(value)
    int_to_name[i] = key
    i += 1
for i in range(len(lst_strands)):
    lst_strands[i] = lst_strands[i][:minLen]
out = join(lst_strands)
for key in int_to_name.keys():
    out = out.replace(str(key) + ":", str(int_to_name[key]) + ":")
draw(out)

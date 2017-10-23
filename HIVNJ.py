from scipy.io import loadmat
from NJ import join
data = loadmat('flhivdata.mat')
lst_strands = []
minLen = 9999999999
for key, value in data.items():
    if "__" in key:
        continue
    value = str(value[0])
    if (len(value) < minLen):
        minLen = len(value)
    lst_strands.append(value)
for i in range(len(lst_strands)):
    lst_strands[i] = lst_strands[i][:minLen]
print(join(lst_strands))

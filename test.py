import numpy as np

array = np.array([[1],[2],[3]])
array2 = np.array([[1],[2]])

array2 = np.insert(array2, [0]*3, array2[0], axis=0)
print(array2)
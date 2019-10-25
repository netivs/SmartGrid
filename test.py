import numpy as np

array = np.array([[1],[2],[3]])
array2 = np.array([[1],[2]])


array = np.vstack((array, array2))
print(array.shape)
print(array)
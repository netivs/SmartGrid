import numpy as np

array = np.array([[1,2,3],[3,4,4]])
array2 = np.array([[5,6,2],[7,8,1]])

array = np.expand_dims(array, axis=2)
array2 = np.expand_dims(array2, axis=2)
array3 = np.concatenate([array,array2], axis=2)
print(array3.shape)
print(array3[0].shape)
print(array3[0])
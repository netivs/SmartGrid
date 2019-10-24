import numpy as np

array = np.arange(20).reshape(4,5)

for i in range(5):
    x = array[:,i]
    x = x.reshape(1, len(x), 1)

arr = np.array(["Hi", "Hello", "Bonjour", "Hola"])

arr.push(var[0:3])
print(arr)
import numpy as np
from datetime import datetime

now = datetime.now()
print("now =", now)
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)	

array = np.array([[[1,2],[1,3]],[[1,4],[0,0]],[[0,0],[0,0]]])
array2 = np.array([[5,6,2],[7,8,1]])
# print(r[~np.all(r == 0, axis=1)])
# array = np.expand_dims(array, axis=2)
# array2 = np.expand_dims(array2, axis=2)
# array3 = np.concatenate([array,array2], axis=2)
List = ['G','E','E','K','S','F', 
        'O','R','G','E','E','K','S'] 
print("Intial List: ") 
print(array[0:2,...])
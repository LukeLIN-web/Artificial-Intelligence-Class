# import numpy as np
# list_three = [[0 for i in range(3)] for j in range(3)]
# print(list_three)
# rect = 2
# rec =3
# roi = list_three[rect:rect ,rec:rec ]
# print(roi)
import numpy as np
# 计算两点之间的距离
def eucliDist(A,B):
    return  abs(np.sqrt(np.sum(np.power((feature - template), 2))))
x = [1.0,2.0,3.0,4.0]
feature =  np.asarray(x)
y =  [0.0,1.1,2.2,3.3]
template = np.asarray(y)
print(eucliDist(feature,template))

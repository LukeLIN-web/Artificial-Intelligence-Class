# import numpy as np
# list_three = [[0 for i in range(3)] for j in range(3)]
# print(list_three)
# rect = 2
# rec =3
# roi = list_three[rect:rect ,rec:rec ]
# print(roi)
import math
# 计算两点之间的距离
def eucliDist(A,B):
    return math.sqrt(sum([(a - b)**2 for (a,b) in zip(A,B)]))
X = [1,2,3,4]
Y = [0,1,2,3]
print(eucliDist(X,Y))
print(len(X))
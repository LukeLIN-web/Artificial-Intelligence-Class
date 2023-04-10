# coding=utf-8
import string
import os
import numpy as np
import matplotlib.pyplot as plt

x = []
y = []

for root, dirs, files in os.walk("./data/"):  # os.walk会该目录下的所有文件
    print(files)
    for file in files:
        name = file.split("_")
        score = float(name[0])
        xidx = int((name[1].split("."))[0])
        x.append(xidx)
        y.append(score)
        print(score)
    plt.figure(figsize=(10, 10), dpi=100)
    plt.scatter(x, y)
    plt.show()



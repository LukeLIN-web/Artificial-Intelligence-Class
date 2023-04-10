# README

This is the guide file of the 2nd project of *Artificial Intelligence*, ISEE, ZJU, 2021 Spring. In this project, you are going to implement a particle filtering algorithm on a object tracking task.

**ATTENTION**: The due of this project is 23:59, 2021.5.17!

## Quick Start

Follow these steps to avoid missing important information:

1. Read `Inroduction.pdf` to preview the project requirements and prerequisites.
2. Read codes under `src/` to get familiar with the definitions and data structures.
3. Complete your codes and check the performance with given datasets `David2` and `car`.
4. Write a brief report of your work. The report can contain the brief introduction of your algorithm,  your innovative ideas and optimizations.

## Format

The work directory should be arranged into:

```
project_2/
|
| --- data/
|
| --- src/
|	  | --- particle_filter_class.py
|	  | --- particle_filter_tracker_main.py
|
| --- report.pdf
```

then upload your project in a compressed package with your name and student ID. E.g. `12345678_张三.zip` . Ensure that your solution codes can run as the descriptions in `Introduction.pdf`.

采样的方式, 对连续的过程滤波处理, 

一开始手动初始化target region

提取特征, 可以改成sift,hog

weighting step  可以用 余弦相似度, 欧氏距离,minkowski距离, 

PCA主成分分析, 



#### resample


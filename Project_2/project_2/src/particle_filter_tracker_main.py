# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:26:54 2019
@author: lee
"""

from path import Path
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import numpy as np
import cv2

from particle_filter_class import Rect, Particle
from particle_filter_class import extract_feature, transition_step, weighting_step, resample_step


def show_img_with_rect(dst_img, rect=None, frame_id=None, particles=None, save_dir=None):
    tmp_img = np.array(dst_img)
    if len(tmp_img.shape) == 2:
        tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_GRAY2BGR)

    if rect is not None:
        cv2.rectangle(tmp_img, (rect.ly, rect.lx), (rect.ly + rect.h, rect.lx + rect.w), (255, 0, 0), 1)
    if frame_id is not None:
        text = 'Frame: {:03d}'.format(frame_id)
        cv2.putText(tmp_img, text, (rect.ly, rect.lx), cv2.FONT_HERSHEY_DUPLEX,
                    1, (0, 0, 255), 2)

    if particles is not None:
        if not isinstance(particles, list):  # Single particle
            particles = [particles]
        for particle in particles:
            cv2.circle(tmp_img, (int(particle.cy), int(particle.cx)), 1, (0, 255, 0), -1)
    #    plt.imshow(tmp_img)
    #    plt.axis('off')
    #    plt.show()
    cv2.imshow('img', tmp_img)
    cv2.waitKey(50)

    if save_dir is not None:
        cv2.imwrite(save_dir / '{:03d}.png'.format(frame_id), tmp_img)


def main():
    home_dir = Path()
    print('Current path: ', home_dir.getcwd())

    #############################################
    # Choose the dataset for evaluation
    #############################################
    test_split = 'car'
    #test_split = 'David2'
    dataset_dir = home_dir / '..' / 'data' / test_split / 'imgs'
    save_dir = home_dir / '..' / 'data' / test_split / 'results'

    # Determine the initial bounding box locationtest_split: 评估数据集名称，可选的为’car’和’David2’，前者的难度较低
    if test_split == 'car':
        init_rect = Rect(68, 47, 97, 115)  # 一开始找一个矩形作为模板
    elif test_split == 'David2':
        # init_rect = Rect(140, 62, 70, 36)
        init_rect = Rect(140, 62, 60, 40)

    #############################################
    # Some variables you can change
    #############################################
    ref_wh = [15, 15]  # Reference size of particle
    # 在预先实现的特征提取函数中，使用像素强度时
    # 为了保证特征向量尺寸一致，我们统一将rect中的图像区域resize到与ref_wh一致的尺寸
    sigmas = [4, 4, 0.03, 0.03]  # Transition sigma of each attr of a particle粒子cx,cy,sx,sy的状态转移标准差
    n_particles = 400  # Number of particles used in particle filter使用的粒子总数，值越高算法速度越慢，但是跟踪性能会越好
    # n_particles = 100
    # feature_type = 'intensity'  # Default feature type, you can try some better features(e.g: HOG)使用的特征类型
    # feature_type = 'histogram'
    feature_type = 'correlation'
    step = 1  # Gap of 读取图像序列的间隔，step=1时，会连续读取图像帧，step=2时，会隔一帧读取图像。该值越高，跟踪的难度越大

    # Read image sequences
    img_list = dataset_dir.files()
    n_imgs = len(img_list)
    if n_imgs < 10:
        print("Too short img sequences length ({}) for tracking ! ".format(n_imgs))
        return

    # Initial particles
    init_img = cv2.imread(img_list[0], -1)
    init_particle = init_rect.to_particle(ref_wh, sigmas=sigmas)
    particles = [init_particle.clone() for i in range(n_particles)]  # Initialize particles
    # 你也可以使用一组4 x N（或N x 4）的 numpy.array来直接表示一组粒子，以便于批量的操作计算

    # Initial matching template 根据 init_rect 提取一个模板出来
    init_features = extract_feature(init_img, init_rect, ref_wh, feature_type)
    template = init_features  # Use feature of latest frame as the matching template

    show_img_with_rect(init_img, init_rect, 0, particles, save_dir=save_dir)

    for idx in range(1, n_imgs, step):
        curr_img = cv2.imread(img_list[idx], -1)

        # Transition particles by Gaussian Distribution
        # 对粒子进行状态转移，预测当前帧的粒子分布
        particles = transition_step(particles, sigmas)

        # Compute each particle's weight by its similarity of template
        # 函数根据每个粒子与匹配模板的相似度来确定其权重
        weights = weighting_step(curr_img, particles, ref_wh, template, feature_type)

        # Decision of best bounding box location by weights
        # 根据每个粒子的权重和分布来确定当前目标的位置，以及生成下一帧的匹配模板。
        # 这里我们预设了一个最简单的策略：使用权重最大的粒子作为当前帧的跟踪结果，其对应的特征向量就是下一帧的匹配模板
        max_idx = np.argmax(weights)
        curr_particle = particles[max_idx]
        template = extract_feature(curr_img, curr_particle.to_rect(ref_wh), ref_wh, feature_type)
        # A vector of features value
        show_img_with_rect(curr_img, curr_particle.to_rect(ref_wh), idx, particles, save_dir=save_dir)

        # Resample the particles by their weights 根据粒子的权重，进行重新采样分配
        particles = resample_step(particles, weights, rsp_sigmas=[2, 2, 0.015, 0.015])
    cv2.destroyAllWindows()


main()

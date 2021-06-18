# #################################################################
# Title: Demo for Tensorflow Usage
# Author: Jiachen Li
# Time: 2021.5.12
# 
# Tips:
# - Tensorflow version: 2.0.0-beta0
# - Codes should also work in any tf2.x versions.
# - This demo is written with static graph paradigm with tf.keras,
#   for dynamic graph implementations, please read official document.
###################################################################

import tensorflow as tf
import numpy as np
import os, argparse

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='', type=str, help='GPU IDs. E.g. 0,1,2. Do not add this argument if you are using CPU mode.')
args = parser.parse_args()

# Assign GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# Simple demo network
class SimpleNN(tf.keras.layers.Layer):
    '''A simple fully-connected neural network demo.'''
    def __init__(self, emb_size, num_class):
        super(SimpleNN, self).__init__()
        self.emb_size = emb_size
        self.num_class = num_class

        # FC layers
        self.fc1 = tf.keras.layers.Dense(self.emb_size, activation=tf.keras.activations.relu)
        self.fc2 = tf.keras.layers.Dense(self.emb_size, activation=tf.keras.activations.relu)
        self.fc3 = tf.keras.layers.Dense(self.emb_size, activation=tf.keras.activations.relu)
        self.fc4 = tf.keras.layers.Dense(self.num_class, activation=tf.keras.activations.softmax)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

def data_preprocess(data):
    h, w = data.shape[1], data.shape[2]
    data = data / 255.0                          # [0,255] -> [0,1]
    data = np.reshape(data, (-1, h*w))           # Squeeze into a 1-dim vector
    data -= np.mean(data, axis=1, keepdims=True) # [0,1] -> [-1,1]
    return data.astype(np.float32)


if __name__ == '__main__':
    # Prepare data
    # MNIST is a hand-written digit dataset with grayscale images ranging from 0-9, pixel value in [0,255].
    # Dataset is splitted into training set and test set, with 60k and 10k samples, respectively.
    # We can call tensorflow.keras.mnist.load_data to get the dataset directly.
    # If you can't connect to storage.google, use the given mnist.npz, place it in ~/.keras/datasets/
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data('./mnist.npz')

    # Preprocessing
    x_train, x_test, x_val = list(map(data_preprocess, [x_train[:x_train.shape[0]//2,:,:], x_test, x_train[x_train.shape[0]//2:,:,:]]))
    y_train, y_val = y_train[:y_train.shape[0]//2], y_train[y_train.shape[0]//2:]

    # Create computation graph
    inputs = tf.keras.layers.Input(shape=(28*28,))
    net = SimpleNN(emb_size=128, num_class=10)
    out = net(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=out)
    opt = tf.keras.optimizers.SGD(learning_rate=0.001)
    model.compile(opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    # Training
    model.fit(x_train, y_train, batch_size=128, epochs=50, validation_data=(x_val, y_val))

    print('------------------------------------------------------')

    # Evaluation
    model.evaluate(x_test, y_test, batch_size=128)
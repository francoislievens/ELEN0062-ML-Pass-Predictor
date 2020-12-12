import numpy as np

import tensorflow as tf

x = [[0.4, 0.8, 0.2, 0.7], [0.4, 0.5, 0.8, 0.9]]

y = tf.nn.softmax(x, axis=1)

print(y.numpy())

targets = [1, 0]

obj = tf.keras.losses.SparseCategoricalCrossentropy()

loss = obj(targets, y)

print(loss)
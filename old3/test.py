
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from Dataset import Dataset
from Neural import Neural
import tensorflow as tf


pred = [0.89, 0.39, 0.8, 0.8, 0.9]
target = [1, 1, 1, 0, 1]

error_obj = tf.keras.losses.BinaryCrossentropy()

err = error_obj(target, pred)

print(err)


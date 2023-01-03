import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress errors

matplotlib.use("TkAgg")
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

trainImage = np.load("dataset/k49-train-imgs.npz")["arr_0"]
trainLabel = np.load("dataset/k49-train-labels.npz")["arr_0"]
testImage = np.load("dataset/k49-test-imgs.npz")["arr_0"]
testLabel = np.load("dataset/k49-test-labels.npz")["arr_0"]

classMap = pd.read_csv("dataset/k49_classmap.csv")

trainImage = trainImage / 255
testImage = testImage / 255

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(49, activation="softmax"),

    ])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


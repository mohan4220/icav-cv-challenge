import cv2
import pickle
import pandas as pd
import numpy as np
import os
import xml.etree.ElementTree as et
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard


x = pickle.load(open("x.pickle", 'rb'))
y = pickle.load(open("y.pickle", "rb"))

print(x.shape, y.shape)

tb = TensorBoard(log_dir="logs")
model = Sequential()
model.add(tf.keras.layers.experimental.preprocessing.Rescaling(1./255))
model.add(Conv2D(64, (3, 3), input_shape=x.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# # =============================================================
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",
              optimizer="adam", metrics=["accuracy"])

history = model.fit(x, y, batch_size=32,
                    validation_split=0.3, epochs=5, callbacks=[tb])

model.save("kangaroo_cnn3.model")

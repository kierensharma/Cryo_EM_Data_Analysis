from numpy.random import seed
from tensorflow.random import set_seed
seed(0)
set_seed(0)

from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from image_data_generation import make_dataset

# data = make_dataframe()
# x = data['Image']
# y = data['Label']

x, y = make_dataset()
x, y = shuffle(x, y)

num_classes = len(np.unique(y))
num_features = x.shape[1]

y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

model = Sequential()
model.add(Dense(128, input_dim=num_features, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.0001), metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=100)

loss, accuracy = model.evaluate(x_test, y_test)

fig = plt.figure()

plt.subplot(1,2,1)
plt.plot(history.history['loss'])
plt.xlabel('epochs')
plt.ylabel('loss')

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'])
plt.xlabel('epochs')
plt.ylabel('accuracy')

plt.show()
# imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import keras
from keras.models import Sequential
from keras.layers import Conv2D, Lambda, MaxPooling2D # convolution layers
from keras.layers import Dense, Dropout, Flatten # core layers
from keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.utils import to_categorical

import warnings
warnings.filterwarnings("ignore")

# Load the data
train = pd.read_csv("../dataset/train.csv")
test = pd.read_csv("../dataset/test.csv")
sub = pd.read_csv('../dataset/sample_submission.csv')

X = train.drop(labels = ['label'], axis = 1).values
y = train['label'].values

X = X.astype('float32')
X = X / 255.0

X = X.reshape(-1,28,28,1)

# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
y = to_categorical(y.astype(np.int32), num_classes = 10)

print(f"Label size {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# print("X_train shape:", X_train.shape)
# print("X_test shape:", X_test.shape)
# print("y_train shape:", y_train.shape)
# print("y_test shape:", y_test.shape)

def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap='binary')
    plt.axis('off')

#Figure size
# plt.figure(figsize=(10, 5))

#CountPlot
# sns.countplot(x='label', data=train)
# plt.title("Distribution of Digits in Training Set")
# plt.show()

X_train__ = X_train.reshape(X_train.shape[0], 28, 28)

# fig, axis = plt.subplots(5, 5, figsize=(22, 10))

# for i, ax in enumerate(axis.flat):
#     ax.imshow(X_train__[i].reshape(28, 28), cmap='binary')
#     digit = y_train[i].argmax()
#     ax.set_title(f"Real Number is: {digit}")

# plt.show()

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
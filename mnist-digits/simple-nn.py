import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score

train = pd.read_csv('../dataset/mnist_train.csv')
test = pd.read_csv('../dataset/mnist_test.csv')

train_labels = train.iloc[:, 0]
train_images = train.iloc[:, 1:]
test_labels = test.iloc[:, 0]
test_images = test.iloc[:, 1:]

train_images = np.asarray(train_images)
train_images = np.reshape(train_images, (60000, 28, 28))
train_images = train_images / 255.0

test_images = np.asarray(test_images)
test_images = np.reshape(test_images, (10000, 28, 28))
test_images = test_images / 255.0

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# history = network.fit(train_images, train_labels, epochs=5, batch_size=128)
# plt.plot(history.history['accuracy'])
# plt.show()

#calculate metrics on test data to see how good the model is performing
metrics = network.evaluate(test_images, test_labels)

# Calculate F1 score
test_probs = network.predict(test_images)
test_predictions = np.argmax(test_probs, axis=1)
test_labels_argmax = np.argmax(test_labels, axis=1)
f1 = f1_score(test_labels_argmax, test_predictions, average='weighted')

print("Test loss:", metrics[0])
print("Test accuracy:", metrics[1])
print("F1 score:", f1)

#using the model to predict values from the test data
predictions = network.predict(test_images[0:10, :])
print("Predictions:", predictions)
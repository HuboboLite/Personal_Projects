import seaborn as sns
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from keras.layers import Dense, Conv2D
from keras.optimizers import Adam
import os
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

train_img = []
train_labels = []

test_img = []
test_labels = []

path_train = '/Users/hubery/PycharmProjects/Brain_Identification_Project/archive (2)/Training/'
path_test = '/Users/hubery/PycharmProjects/Brain_Identification_Project/archive (2)/Testing/'
img_size = 300

for i in os.listdir(path_train):
    for j in os.listdir(path_train + i):
        train_img.append(cv2.resize(cv2.imread(path_train + i + '/' + j), (img_size, img_size)))
        train_labels.append(i)

for i in os.listdir(path_test):
    for j in os.listdir(path_test + i):
        test_img.append(cv2.resize(cv2.imread(path_test + i + '/' + j), (img_size, img_size)))
        test_labels.append(i)

train_img = (np.array(train_img))
test_img = (np.array(test_img))

train_labels_encoded = [
    0 if category == 'no_tumor' else (1 if category == 'glioma_tumor' else (2 if category == 'meningioma_tumor' else 3))
    for category in list(train_labels)]
test_labels_encoded = [
    0 if category == 'no_tumor' else (1 if category == 'glioma_tumor' else (2 if category == 'meningioma_tumor' else 3))
    for category in list(test_labels)]

print("Shape of train: ", (train_img).shape, " and shape of test: ", (test_img).shape)

img_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True)

img_datagen.fit(train_img)
img_datagen.fit(test_img)

train_x, val_x, train_y, val_y = train_test_split(np.array(train_img), np.array(train_labels), test_size=0.1)
# print(train_x.shape, train_y.shape, val_x.shape, val_y.shape)


plt.figure(figsize=(15, 15))
for i, j in enumerate(train_img):
    if i < 5:
        plt.subplot(1, 5, i + 1)
        plt.imshow(j)
        plt.xlabel(train_labels[i])
        plt.tight_layout()
    else:
        break

plt.figure(figsize=(17, 8));
lis = ['Train', 'Test']
for i, j in enumerate([train_labels, test_labels]):
    plt.subplot(1, 2, i + 1);
    sns.countplot(x=j);
    plt.xlabel(lis[i])


model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(kernel_size=(5, 5), filters=32, activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=32, activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=32, activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=64, activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(4, activation='sigmoid')
    ])
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(tf.cast(train_img, tf.float32), np.array(pd.get_dummies(train_labels)), validation_split=0.1,
                    epochs=20, verbose=1, batch_size=32)

# print(model.layers[0].get_weights()[0].shape)

train_x, val_x, train_y, val_y = train_test_split(np.array(train_img), np.array(train_labels), test_size=0.1)
# print(train_x.shape, train_y.shape, val_x.shape, val_y.shape)


keras.models.save_model(model, 'Brain_Identification.keras', overwrite=False)

def preprocess_image(image_path):
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# Randomly select 4 images from the training set
random_indices = np.random.choice(len(test_img), size=4, replace=False)
random_images = [test_img[i] for i in random_indices]

# Make predictions on the selected images
plt.figure(figsize=(12, 6))
for i, image_path in enumerate(random_images):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction[0])
    class_mapping = {0: 'no_tumor', 1: 'glioma_tumor', 2: 'meningioma_tumor', 3: 'pituitary_tumor'}
    predicted_label = class_mapping[predicted_class]

    plt.subplot(1, 4, i + 1)
    plt.imshow(cv2.imread(image_path))
    plt.title(f"Prediction: {predicted_label}")
    plt.axis('off')

plt.show()

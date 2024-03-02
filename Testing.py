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

model = keras.models.load_model("/Users/hubery/PycharmProjects/Brain_Identification_Project/Brain_Identification.keras")

def preprocess_single_image(image_path):
    try:
        img = cv2.imread(str(image_path))
        if img is None or img.size == 0:
            raise ValueError("Error: Image not loaded or empty.")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)
    except Exception as e:
        print(f"Error loading image: {image_path}")
        print(e)
        return None

# Test the model using specific images
import random

image_paths = [
    '/Users/hubery/PycharmProjects/Brain_Identification_Project/archive (2)/Testing/no_tumor/image(1).jpg',
    '/Users/hubery/PycharmProjects/Brain_Identification_Project/archive (2)/Testing/glioma_tumor/image(1).jpg',
    '/Users/hubery/PycharmProjects/Brain_Identification_Project/archive (2)/Testing/meningioma_tumor/image(1).jpg',
    '/Users/hubery/PycharmProjects/Brain_Identification_Project/archive (2)/Testing/pituitary_tumor/image(1).jpg'
]

folder_paths = [
    '/Users/hubery/PycharmProjects/Brain_Identification_Project/archive (2)/Testing/no_tumor/',
    '/Users/hubery/PycharmProjects/Brain_Identification_Project/archive (2)/Testing/glioma_tumor/',
    '/Users/hubery/PycharmProjects/Brain_Identification_Project/archive (2)/Testing/meningioma_tumor/',
    '/Users/hubery/PycharmProjects/Brain_Identification_Project/archive (2)/Testing/pituitary_tumor/'
]


# Function to check if an image path is already in the list
def is_duplicate_path(path, path_list):
    return path in path_list


# Randomly select 4 images from each folder until we have 16 unique images
while len(image_paths) < 16:
    folder_path = random.choice(folder_paths)
    image_files = os.listdir(folder_path)
    random_image = random.choice(image_files)
    image_path = os.path.join(folder_path, random_image)
    if not is_duplicate_path(image_path, image_paths):
        image_paths.append(image_path)



plt.figure(figsize=(16, 8))
for i, image_path in enumerate(image_paths):
    preprocessed_image = preprocess_single_image(image_path)
    if preprocessed_image is None:
        continue

    prediction = model.predict(preprocessed_image)
    print(prediction)
    predicted_class = np.argmax(prediction[0])
    class_mapping = {0: 'no_tumor', 1: 'glioma_tumor', 2: 'meningioma_tumor', 3: 'pituitary_tumor'}
    predicted_label = class_mapping[predicted_class]

    plt.subplot(2, 8, i + 1)
    plt.imshow(cv2.imread(image_path))
    plt.title(f"Prediction: {predicted_label}")
    plt.axis('off')

plt.show()



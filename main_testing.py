import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import torch
import torchvision
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import os
from pathlib import Path
import time
import numpy as np
import cv2
from keras.models import model_from_json
from roboflow import Roboflow
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import matplotlib.image as mpimg
from roboflow import Roboflow
rf = Roboflow(api_key="LTbPR7zAynPXC51hOJIW")
project = rf.workspace().project("braintumordetection-hxkiz")
object = project.version(2).model

# Define your model architecture
img_size = 300

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc1 = nn.Linear(360000, 128)  # Adjusted input size
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # print(x.size(1))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x



# Load the model and classes from cifar_net2.pth
model_path = '/Users/hubery/PycharmProjects/Brain_Identification_Project/cifar_net.pth'  # Replace with the path to your model file
checkpoint = torch.load(model_path)

# Create an instance of your model
model = Net()

# Load the model state_dict from the checkpoint
model.load_state_dict(checkpoint)

# Set the model to evaluation mode
model.eval()

# Load the classes
classes = ('glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor')


image_path = '/Users/hubery/PycharmProjects/Brain_Identification_Project/dd86e8296503e5713851c6908df2bd_gallery.jpg'


# Prepare the image for prediction
image = Image.open(image_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_image = transform(image).unsqueeze(0)

# Make the prediction
with torch.no_grad():
    output = model(input_image)
    # print(output)
    _, predicted_idx = torch.max(output, 1)
    predicted_class = classes[predicted_idx.item()]
    print("Predicted class:", predicted_class)

# visualize your prediction
object.predict(image_path, confidence=30, overlap=30).save("prediction.jpg")

img = mpimg.imread('prediction.jpg')
imgplot = plt.imshow(img)
plt.show()

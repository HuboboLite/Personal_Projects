import seaborn as sns
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

class BrainDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img = cv2.imread(self.image_paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        label = self.labels[index]
        return img, label

train_img = []
train_labels = []

test_img = []
test_labels = []

path_train = '/Users/hubery/PycharmProjects/Brain_Identification_Project/archive (2)/Training/'
path_test = '/Users/hubery/PycharmProjects/Brain_Identification_Project/archive (2)/Testing/'
img_size = 300

for i in os.listdir(path_train):
    for j in os.listdir(path_train + i):
        train_img.append(os.path.join(path_train, i, j))
        train_labels.append(i)

for i in os.listdir(path_test):
    for j in os.listdir(path_test + i):
        test_img.append(os.path.join(path_test, i, j))
        test_labels.append(i)

train_labels_encoded = [0 if category == 'no_tumor' else (1 if category == 'glioma_tumor' else (2 if category == 'meningioma_tumor' else 3))
                        for category in train_labels]
test_labels_encoded = [0 if category == 'no_tumor' else (1 if category == 'glioma_tumor' else (2 if category == 'meningioma_tumor' else 3))
                       for category in test_labels]

print("Shape of train: ", len(train_img), " and shape of test: ", len(test_img))

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

train_dataset = BrainDataset(train_img, train_labels_encoded, transform=transform)
test_dataset = BrainDataset(test_img, test_labels_encoded, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * (img_size // 16) * (img_size // 16), 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 64 * (img_size // 16) * (img_size // 16))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = Net().to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

def train(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    corrects = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)
    val_loss = running_loss / len(dataloader.dataset)
    val_acc = corrects.double() / total
    return val_loss, val_acc

num_epochs = 20
train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer)
    val_loss, val_acc = evaluate(model, test_loader, criterion)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc.item())
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

num_epochs = 20
train_losses = []
val_losses = []
val_accuracies = []

# Code to take 4 random images from the training set
random_indices = np.random.choice(len(train_dataset), size=4, replace=False)
random_images = [train_dataset[i][0] for i in random_indices]
random_labels = [train_dataset[i][1] for i in random_indices]

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer)
    val_loss, val_acc = evaluate(model, test_loader, criterion)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc.item())
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

# Make predictions on the 4 random images
model.eval()
with torch.no_grad():
    random_images = torch.stack(random_images).to(device)
    outputs = model(random_images)
    _, predicted_labels = torch.max(outputs, 1)

# Map predicted labels to actual classes
class_mapping = {0: 'no_tumor', 1: 'glioma_tumor', 2: 'meningioma_tumor', 3: 'pituitary_tumor'}
predicted_classes = [class_mapping[label.item()] for label in predicted_labels]

# Display predictions
plt.figure(figsize=(12, 6))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(random_images[i].cpu().permute(1, 2, 0))
    plt.title(f"Predicted: {predicted_classes[i]}\nTrue: {class_mapping[random_labels[i]]}")
    plt.axis('off')

plt.show()

torch.save(model.state_dict(), 'Brain_Identification.pth')
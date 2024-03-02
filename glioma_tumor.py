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
import torch.optim.lr_scheduler as lr_scheduler
import time


start = time.perf_counter()

img_size = 128

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
testing = '/Users/hubery/PycharmProjects/Brain_Identification_Project/archive (2) copy/Testing'
training = '/Users/hubery/PycharmProjects/Brain_Identification_Project/archive (2) copy/Training'

batch_size = 4

training_data = torchvision.datasets.ImageFolder(root=training, transform=transform)
train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=0)
testing_data = torchvision.datasets.ImageFolder(root=testing, transform=transform)
test_loader = torch.utils.data.DataLoader(testing_data, batch_size=batch_size, shuffle=False, num_workers=0)
accuracy_loader = torch.utils.data.DataLoader(testing_data, batch_size=400, shuffle=False, num_workers=0)

classes = ('glioma_tumor', 'no_glioma')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    npimg = np.clip(npimg, 0, 1)  # clip values to [0, 1]
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

'''
# get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# show images
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

imshow(torchvision.utils.make_grid(images))
'''


class Net(nn.Module):
    def __init__(self, in_channels, num_diseases):
        super().__init__()

        self.conv1 = self.ConvBlock(in_channels, 64)
        self.conv2 = self.ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(self.ConvBlock(128, 128), self.ConvBlock(128, 128))

        self.conv3 = self.ConvBlock(128, 256, pool=True)
        self.conv4 = self.ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(self.ConvBlock(512, 512), self.ConvBlock(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(2),
                                        nn.Flatten(),
                                        nn.Linear(512, num_diseases))

    @staticmethod
    def ConvBlock(in_channels: int, out_channels: int, pool: bool = False) -> nn.Sequential:
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace=True)]
        if pool:
            layers.append(nn.MaxPool2d(4))
        return nn.Sequential(*layers)

    def forward(self, batch):
        out = self.conv1(batch)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


print('Working...')


net = Net(3, 2)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.00001)
# Earlier learning rate: 0.0001 - 75% accuracy with Adam optimizer
print('Training...')


epochs = 10

accuracy_list = []
learning_rate_store = []
learning_rate = 0.001

optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.00001)
initial_lr = 0.001
lr_factor = 0.1

# Create the StepLR scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=lr_factor)

for epoch in range(epochs):  # loop over the dataset multiple times
    print('Epoch: ' + str(epoch+1))
    running_loss = 0.0
    loss_total = 0.0
    for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # print statistics
        prev_loss = running_loss
        running_loss += loss.item()
        print(f'\nLoss of iteration {i+1}: {running_loss/(i+1)} \nChange --> {running_loss/(i+1)-prev_loss/(i if i>=1 else 1)}')
        # loss_total += running_loss/(i+1)-prev_loss/(i if i >= 1 else 1)
        # print(f'Loss total: {loss_total}')
        '''
        if i % 99 == 0:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 1000:.3f}')
            running_loss = 0.0
        '''
    if epoch == 0:
        print(f'Seconds passed: {time.perf_counter() - start}')
        print(f'Minutes needed: {(time.perf_counter() - start) * epochs // 60}')

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for count, data in enumerate(accuracy_loader):
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the {len(accuracy_loader)} test images: {100 * correct // total} %')
    accuracy_list.append(f'Epoch {epoch+1}: {100*correct // total}')

print('Finished Training')
print(accuracy_list)
print(learning_rate_store)

PATH = './cifar_net3.pth'
torch.save(net.state_dict(), PATH)

dataiter = iter(test_loader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

PATH = './cifar_net3.pth'
net = Net(3, 2)
net.load_state_dict(torch.load(PATH))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')



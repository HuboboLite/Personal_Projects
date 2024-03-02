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
import torch.optim.lr_scheduler as lr_scheduler
from skorch import NeuralNetClassifier


start = time.perf_counter()

img_size = 300

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
training = 'file_path_for_training'
testing = 'file_path_for_testing'

batch_size = 4

training_data = torchvision.datasets.ImageFolder(root=training, transform=transform)
train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=0)
testing_data = torchvision.datasets.ImageFolder(root=testing, transform=transform)
test_loader = torch.utils.data.DataLoader(testing_data, batch_size=batch_size, shuffle=False, num_workers=0)
accuracy_loader = torch.utils.data.DataLoader(testing_data, batch_size=400, shuffle=False, num_workers=0)

classes = ('glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    npimg = np.clip(npimg, 0, 1)  # clip values to [0, 1]
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

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




print('Working...')

model_path = '/Users/hubery/PycharmProjects/Brain_Identification_Project/cifar_net2.pth'  # Replace with the path to your model file
checkpoint = torch.load(model_path)

# net = Net(3, 4)
net = Net()
net.load_state_dict(checkpoint)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.00001)
# Earlier learning rate: 0.0001 - 75% accuracy with Adam optimizer
print('Training...')


epochs = 2

accuracy_list = []
learning_rate_store = []
learning_rate = 0.001

optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.00001)
initial_lr = 0.0001
lr_factor = 0.01

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

PATH = './cifar_net2.pth'
torch.save(net.state_dict(), PATH)

dataiter = iter(test_loader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

PATH = './cifar_net2.pth'
net = Net()
# Net(3, 4)
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




'''
# get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# show images
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

imshow(torchvision.utils.make_grid(images))
'''

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
'''

'''
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


class Net(nn.Module):
    def __init__(self, in_channels, num_diseases):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)  # out_dim : 128 x 64 x 64
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))

        self.conv3 = ConvBlock(128, 256, pool=True)  # out_dim : 256 x 16 x 16
        self.conv4 = ConvBlock(256, 512, pool=True)  # out_dim : 512 x 4 x 44
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Linear(512, num_diseases))

    def forward(self, xb):  # xb is the loaded batch
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

'''


"""
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=1)
        self.bn32 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=1)
        self.bn64 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 4, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 4, padding=1)
        self.bn128 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc1 = nn.Linear(64, 128)  # Adjusted input size based on the output shape after convolutions
        self.fc2 = nn.Linear(128, 4)
        self.fc3 = nn.Linear(4, 32)
        self.fc4 = nn.Linear(32, 4)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn32(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(self.bn64(self.conv2(x)))
        x = self.pool(self.bn128(self.conv3(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn128(self.conv4(x))))
        x = self.pool(self.bn64(self.conv5(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn64(self.conv6(x))))
        x = self.dropout(x)

        # Apply average pooling
        x = F.avg_pool2d(x, x.size()[2:])

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # print(x.size(1))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
"""

'''
class Net(nn.Module):
    def __init__(self, num_classes=4):
        super(Net, self).__init__()
        self.vgg16 = models.vgg16(weights= 'VGG16_Weights.DEFAULT')
        # Freeze all the parameters in the VGG-16 model
        for param in self.vgg16.parameters():
            param.requires_grad = False

        # Modify the last fully connected layer to match the number of output classes
        num_features = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.vgg16(x)
        return x
'''

'''
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
'''


'''
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 2048, 5, padding=1)
        self.bn32 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(2048, 1024, 5, padding=1)
        self.bn64 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(1024, 1024, 5, padding=1)
        self.conv4 = nn.Conv2d(1024, 512, 4, padding=1)
        self.bn128 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(512, 512, 4, padding=1)
        self.conv6 = nn.Conv2d(512, 256, 4, padding=1)
        self.conv7 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv8 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv9 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv10 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn256 = nn.BatchNorm2d(256)
        self.bn512 = nn.BatchNorm2d(512)
        self.bn1024 = nn.BatchNorm2d(1024)
        self.bn2048 = nn.BatchNorm2d(2048)
        self.conv11 = nn.Conv2d(32, 32, 3, padding=1)
        self.fc1 = nn.Linear(32, 128)  # Adjusted input size based on the output shape after convolutions
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 4)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn2048(self.conv1(x))))
        x = self.dropout(x)
        x = F.relu(self.bn1024(self.conv2(x)))
        x = self.pool(F.relu(self.bn1024(self.conv3(x))))
        x = self.dropout(x)
        x = F.relu(self.bn512(self.conv4(x)))
        x = self.pool(F.relu(self.bn512(self.conv5(x))))
        x = self.dropout(x)
        x = F.relu(self.bn256(self.conv6(x)))
        x = F.relu(self.bn256(self.conv7(x)))
        x = self.pool(F.relu(self.bn128(self.conv8(x))))
        x = self.dropout(x)
        x = F.relu(self.bn64(self.conv9(x)))
        x = F.relu(self.bn32(self.conv10(x)))
        # print(x.size())
        x = self.pool(F.relu(self.bn32(self.conv11(x))))
        x = self.dropout(x)

        # Apply average pooling
        x = F.avg_pool2d(x, x.size()[2:])

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
'''

'''
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn32 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 4, padding=1)
        self.bn64 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 4, padding=1)
        self.bn128 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv6 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv7 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv8 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv9 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv10 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn256 = nn.BatchNorm2d(256)
        self.bn512 = nn.BatchNorm2d(512)
        self.conv11 = nn.Conv2d(32, 32, 3, padding=1)
        self.fc1 = nn.Linear(32, 64)  # Adjusted input size based on the output shape after convolutions
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 4)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn32(self.conv1(x))))
        x = self.dropout(x)
        x = F.relu(self.bn64(self.conv2(x)))
        x = self.pool(F.relu(self.bn128(self.conv3(x))))
        x = self.dropout(x)
        x = F.relu(self.bn256(self.conv4(x)))
        x = self.pool(F.relu(self.bn512(self.conv5(x))))
        x = self.dropout(x)
        x = F.relu(self.bn512(self.conv6(x)))
        x = F.relu(self.bn256(self.conv7(x)))
        x = self.pool(F.relu(self.bn128(self.conv8(x))))
        x = self.dropout(x)
        x = F.relu(self.bn64(self.conv9(x)))
        x = F.relu(self.bn32(self.conv10(x)))
        x = self.pool(F.relu(self.bn32(self.conv11(x))))
        x = self.dropout(x)

        # Apply average pooling
        x = F.avg_pool2d(x, x.size()[2:])

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # print(x.size(1))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
'''


'''
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)  # Input channels changed to 1
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(128, 256, 3)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv7 = nn.Conv2d(256, 256, 3)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 512, 3)
        self.bn8 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv9 = nn.Conv2d(512, 256, 3)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(256, 128, 3)
        self.bn10 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv11 = nn.Conv2d(128, 64, 3)
        self.bn11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(64, 64, 3)
        self.bn12 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv13 = nn.Conv2d(64, 64, 3)
        self.bn13 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(5326848, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 4)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))
        x = self.bn6(self.conv6(x))
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.bn8(self.conv8(x))
        x = self.bn7(self.conv9(x))
        x = F.relu(self.bn3(self.conv10(x))) # 9: 256, 128, 64, 64, 64
        x = self.bn2(self.conv11(x))
        x = self.bn2(self.conv12(x))
        x = F.relu(self.bn2(self.conv13(x)))
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        print(x.size(1))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x
'''

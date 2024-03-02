import ray
from ray import tune
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


start = time.perf_counter()


transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
training = '/Users/hubery/PycharmProjects/Brain_Identification_Project/archive (2)/Training'
testing = '/Users/hubery/PycharmProjects/Brain_Identification_Project/archive (2)/Testing'

batch_size = 4
img_size = 300

training_data = torchvision.datasets.ImageFolder(root=training, transform=transform)
train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=0)
testing_data = torchvision.datasets.ImageFolder(root=testing, transform=transform)
test_loader = torch.utils.data.DataLoader(testing_data, batch_size=batch_size, shuffle=False, num_workers=0)
accuracy_loader = torch.utils.data.DataLoader(testing_data, batch_size=400, shuffle=False, num_workers=0)

classes = ('glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor')


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

def train_model(config, checkpoint_dir=None):
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=config["learning_rate"], weight_decay=0.00001)

    if checkpoint_dir:
        checkpoint = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    epochs = config["epochs"]
    accuracy_list = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        print('Epoch: ' + str(epoch + 1))
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
            print(
                f'\nLoss of iteration {i + 1}: {running_loss / (i + 1)} \nChange --> {running_loss / (i + 1) - prev_loss / (i if i >= 1 else 1)}')
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

        # Tune checkpointing to keep track of the model at each epoch
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save({
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }, path)

        print(f'Accuracy of the network on the {len(accuracy_loader)} test images: {100 * correct // total} %')
        accuracy_list.append(f'Epoch {epoch + 1}: {100 * correct // total}')

        # Tune reporting for hyperparameter optimization
        tune.report(accuracy=correct//total, loss=running_loss / (i + 1))


# Define the hyperparameter search space
config = {
    "learning_rate": tune.loguniform(7.5e-5, 5e-4),
    "epochs": tune.choice([5, 10, 15]),
}

# Initialize Ray and run the hyperparameter search
ray.init()

analysis = tune.run(
    train_model,
    config=config,
    num_samples=6,  # Number of configurations to try (reduced to 5)
    metric="accuracy",
    mode="max",
)

best_trial = analysis.get_best_trial("accuracy", "max", "last")
print("Best trial config: {}".format(best_trial.config))
print("Best trial final accuracy: {}".format(best_trial.last_result["accuracy"]))
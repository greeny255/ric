from __future__ import print_function, division
from torchvision import transforms
from multiprocessing import freeze_support

import os
import sys
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from SceneClassification import SceneClassificationDataset, Net, Network

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

# check device to use GPU or CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

# setting batch size
BATCH_SIZE = 4

# imgae transform
transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((300,300)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# Load data to train
dataset = SceneClassificationDataset("./hust/", ["food", "indoor", "outdoor"], transform=transform)
# suppose dataset is the variable pointing to whole datasets
N = len(dataset)
# generate & shuffle indices
indices = np.arange(N)
indices = np.random.permutation(indices)
# there are many ways to do the above two operation. (Example, using np.random.choice can be used here too

# select train/test/val, for demo I am using 70,15,15
train_indices = indices [:int(0.7*N)]
train_data = data.Subset(dataset, train_indices)
trainloader = data.DataLoader(train_data,
                                 shuffle=True,
                                 batch_size=BATCH_SIZE)

# Load model
net = Net()
# net = Network()
net.to(device)

# get some random training images
if __name__ == '__main__':
    freeze_support()
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show images
    dataset.imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{dataset.classes[labels[j]]:5s}' for j in range(BATCH_SIZE)))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # Train
    print('Start Training')
    for epoch in range(10):
        # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            data[1] = data[1].type(torch.LongTensor)
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    # save training results
    PATH = './classification_model.pth'
    torch.save(net.state_dict(), PATH)

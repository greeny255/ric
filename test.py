from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from multiprocessing import freeze_support

from PIL import Image
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms.functional as TF

from tqdm.notebook import trange, tqdm
import matplotlib.pyplot as plt
import numpy as np

classes = ['food','indoor','outdoor']

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

mean = 0
std = 0

# transform = transforms.Compose(
#     [transforms.ToPILImage(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# transform
transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((300,300)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
# load image from path
def unpickle(path, size=1000):
    imgs = []
    # valid_images = [".jpg", ".gif", ".png", ".tga"]
    index = 0
    for f in os.listdir(path):
        if index >= size:
            break
        # ext = os.path.splitext(f)[1]
        image = Image.open(os.path.join(path, f))
        x = TF.to_tensor(image)
        x.unsqueeze_(0)
        imgs.append(x[0])
        index += 1
    return imgs

# Data Loader
class Cifar10Dataset(Dataset):
    """Cifar 10 dataset."""

    def __init__(self, root_dir, file_paths, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.root_dir = root_dir
        self.file_paths = file_paths
        self.transform = transform

        self.data, self.labels = self.load_data()

    def __len__(self):
        return len(self.data)

    def load_data(self):
        # load data
        images = []
        label_images = []
        for file_path in self.file_paths:
            _raw_data = unpickle(self.root_dir + file_path)
            images += _raw_data
            # create labels
            _label = classes.index(file_path)
            _raw_labels = [_label for i in range(len(_raw_data))]
            label_images += _raw_labels

        # images = [img.transpose(1, 2, 0) for img in images]

        # self.mean = np.mean(images) / 255
        # self.std = np.std(images) / 255

        return images, label_images


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.data[idx]
        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]

# Load data to train
BATCH_SIZE = 4

dataset = Cifar10Dataset("./hust/", ["food", "indoor", "outdoor"], transform=transform)
# suppose dataset is the variable pointing to whole datasets
N = len(dataset)
# generate & shuffle indices
indices = np.arange(N)
indices = np.random.permutation(indices)
# there are many ways to do the above two operation. (Example, using np.random.choice can be used here too

# select train/test/val, for demo I am using 70,15,15
train_indices = indices [:int(0.7*N)]
val_indices = indices[int(0.7*N):int(0.85*N)]
test_indices = indices[int(0.85*N):]

train_data = data.Subset(dataset, train_indices)
val_data = data.Subset(dataset, val_indices)
test_data = data.Subset(dataset, test_indices)

item1 = train_data.__getitem__(0)

trainloader = data.DataLoader(train_data,
                                 shuffle=True,
                                 batch_size=BATCH_SIZE)
testloader = data.DataLoader(test_data,
                                batch_size=BATCH_SIZE)




# functions to show an image
def imshow(img):
    # img = img / 250     # unnormalize
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Model Net
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(82944, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = x.float()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Model Network
# Define a convolution neural network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24 * 10 * 10, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.pool(output)
        output = F.relu(self.bn4(self.conv4(output)))
        output = F.relu(self.bn5(self.conv5(output)))
        output = output.view(-1, 24 * 10 * 10)
        # output = self.fc1(output)
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)

        return output

# check device to use GPU or CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

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
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(BATCH_SIZE)))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # Train
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

    PATH = './classification_model.pth'
    torch.save(net.state_dict(), PATH)

    # Run Test
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    # Load model
    net = Net()
    net.load_state_dict(torch.load(PATH))

    outputs = net(images)

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
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





"""This contains code for building a CNN to classify single proteins"""
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pdb
from utils import ImageLoader


class Net(nn.Module):

    # def __init__(self):
    #     super(Net, self).__init__()
    #     # 1 input image channel, 6 output channels, 5x5 square convolution
    #     # kernel
    #     self.conv1 = nn.Conv2d(4, 6, 5)
    #     self.conv2 = nn.Conv2d(6, 16, 5)
    #     # an affine operation: y = Wx + b
    #     self.fc1 = nn.Linear(16 * 5 * 5, 120)
    #     self.fc2 = nn.Linear(120, 84)
    #     self.fc3 = nn.Linear(84, 1)
    #
    # def forward(self, x):
    #     # Max pooling over a (2, 2) window
    #     x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
    #     # If the size is a square you can only specify a single number
    #     x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    #     x = x.view(-1, self.num_flat_features(x))
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return x

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(4, 6, 4)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.out = nn.Linear(4, 2)
        # Barley learning, need to beef up model
        # Beefed up by having first layer: 4 to 128, then 128 to 256, bunch of linear down
        self.fc1 = nn.Linear(13456, 124)
        self.fc2 = nn.Linear(124, 24)
        self.fc3 = nn.Linear(24, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(4, -1) # 16 * 4 * 4
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = self.out(x)
        return x

net = Net()
net.zero_grad()

transform = transforms.Compose(
    [transforms.ToTensor()])

trainset = ImageLoader('zika', tform=transform)
# source_dataloader = torch.utils.data.DataLoader(source_dataset, batch_size=32, shuffle=False, drop_last=True, num_workers=15)
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                          shuffle=False, num_workers=2)

# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
for epoch in range(500):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        print(epoch, i)
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        # print(outputs.size())
        # print(labels.size())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 200 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

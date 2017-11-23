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

    def __init__(self, num_prots, batchsize=4, img_chan=4):
        super(Net, self).__init__()
        self.C = num_prots
        self.N = batchsize
        self.conv1 = nn.Conv2d(img_chan, 6, 4)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # self.out = nn.Linear(4, 2)
        # Barley learning, need to beef up model - FIXED by resizing images bug
        # Beefed up by having first layer: 4 to 128, then 128 to 256, bunch of linear down
        self.fc1 = nn.Linear(13456, 124)
        self.fc2 = nn.Linear(124, 24)
        self.fc3 = nn.Linear(24, self.C)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Should probably have not hard code batchsize of 4
        x = x.view(self.N, -1) # 16 * 4 * 4
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = self.out(x)
        return x


def forward_backward(net, crit, optim, data, back=True):
    inputs, labels = data
    inputs, labels = Variable(inputs), Variable(labels)
    optim.zero_grad()
    outputs = net(inputs)
    loss = crit(outputs, labels)
    if back:
        loss.backward()
        optim.step()
    return loss


def train(net, epochs, crit, optim, trainloader, testloader=None):
    """Trains net"""
    if testloader:
        data_gen = zip(trainloader, testloader)
    else:
        data_gen = trainloader

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, zipped in enumerate(data_gen, 0):
            # get the inputs
            if testloader:
                data, test_data = zipped
            else:
                data = zipped

            print(epoch, i)

            loss = forward_backward(net, crit, optim, data)

            if testloader:
                # Test error
                loss = forward_backward(net, crit, optim, test_data, back=False)

            running_loss += loss.data[0]
            if i % 200 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0


net = Net(2)
net.zero_grad()

trainset = ImageLoader(tform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

# Should we still be doing pos / neg?
testset = ImageLoader(tform=transforms.ToTensor(), test=True)

testloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

train(net, 3, criterion, optimizer, trainloader, testloader=testloader)
# for epoch in range(2):  # loop over the dataset multiple times
#
#     running_loss = 0.0
#     for i, zipped in enumerate(zip(trainloader, testloader), 0):
#         # get the inputs
#         data, test_data = zipped
#         print(epoch, i)
#         inputs, labels = data
#
#         # wrap them in Variable
#         inputs, labels = Variable(inputs), Variable(labels)
#
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + backward + optimize
#         outputs = net(inputs)
#         # print(outputs.size())
#         # print(labels.size())
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         # Test error
#         inputs, labels = test_data
#         inputs, labels = Variable(inputs), Variable(labels)
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         running_loss += loss.data[0]
#         if i % 200 == 0:    # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 200))
#             running_loss = 0.0

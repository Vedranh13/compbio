"""This contains code for building a CNN to classify single proteins"""
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from utils import ImageLoader, ImageLoaderMult
from noise import make_noisy_tf, calc_var
from sliding_window import decomp_im


class Net(nn.Module):

    def __init__(self, num_prots, width=124, height=124, img_chan=4):
        super(Net, self).__init__()
        self.C = num_prots
        self.W = width
        self.H = height
        self.CHAN = img_chan
        self.conv1 = nn.Conv2d(img_chan, 6, 4)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # self.out = nn.Linear(4, 2)
        # Barley learning, need to beef up model - FIXED by resizing images bug
        # Beefed up by having first layer: 4 to 128, then 128 to 256, bunch of linear down
        self.fc1 = nn.Linear(13456, 124)
        self.fc2 = nn.Linear(124, 24)
        self.fc3 = nn.Linear(24, self.C)
        self.fcMult = nn.Linear(self.W * self.H * self.C, self.C ** 2)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Should probably have not hard code batchsize of 4
        # x = x.view(self.N, -1) # 16 * 4 * 4
        x = x.view(-1, 13456)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = self.out(x)
        return x

    def multiple(self, im):
        # import pdb; pdb.set_trace()
        w, h = im.size()[1:]
        all_decomps = decomp_im(im, self.W, self.H)
        to_ten = transforms.ToTensor().__call__
        scores = torch.cuda.FloatTensor(w - self.W, h - self.H, self.C)
        for i, prot in enumerate(all_decomps, 0):
            mult = prot.unsqueeze(0)
            # mult = mult.type(torch.cuda.FloatTensor)
            mult = Variable(mult)
            # import pdb; pdb.set_trace()
            scores[i // self.W][i % self.W] = self.forward(mult).data[0]
            # import pdb; pdb.set_trace()
        return scores

    def multiple_batch(self, ims, bs=4):
        stacked = torch.cuda.FloatTensor(bs, self.C, self.W, self.H)
        for i in range(bs):
            stacked[i] = self.multiple(ims[i]).permute(2, 0, 1)
        return stacked

    def multiple_nn(self, ims, bs=4):
        scores = self.multiple_batch(ims, bs=bs)
        scores = scores.view(-1, self.W * self.H * self.C)
        scores = Variable(scores)
        # import pdb; pdb.set_trace()
        return self.fcMult(scores)


def forward_backward(net, crit, optim, data, back=True, cuda=True, mult=False):
    inputs, labels = data
    if cuda:
        inputs, labels = inputs.type(torch.cuda.FloatTensor), labels.type(torch.cuda.LongTensor)
    inputs, labels = Variable(inputs), Variable(labels)
    optim.zero_grad()
    if mult:
        outputs = net.multiple_nn(inputs.data)
    else:
        outputs = net(inputs)
    loss = crit(outputs, labels)
    if back:
        loss.backward()
        optim.step()
    return loss


def train(net, epochs, crit, optim, trainloader, testloader=None, total_err=False, cuda=True, mult=False):
    """Trains net"""
    losses = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        if testloader:
            data_gen = zip(trainloader, testloader)
        else:
            data_gen = trainloader
        running_loss = 0.0
        for i, zipped in enumerate(data_gen, 0):
            # get the inputs
            if testloader:
                data, test_data = zipped
            else:
                data = zipped

            print(epoch, i)

            loss = forward_backward(net, crit, optim, data, mult=mult)

            if testloader:
                # Test error
                loss = forward_backward(net, crit, optim, test_data, back=False, mult=mult)

            running_loss += loss.data[0]
            if i % 200 == 0 and i != 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                losses.append(running_loss / 200)
                running_loss = 0.0
    if not total_err:
        return losses
    loss = 0
    n = 0
    dg = iter(testloader)
    for test_data in dg:
        loss += forward_backward(net, crit, optim, test_data, back=False)
        n += 1
    losses.append(loss / n)
    return losses


def train_simple(report_errs=False):
    net = Net(3)
    net.zero_grad()
    net.cuda()
    tf = transforms.Compose([transforms.ToTensor()])#, transforms.Lambda(lambda x: x.cuda())])
    # tf = transforms.ToTensor()
    trainset = ImageLoader(tform=tf)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    testset = ImageLoader(tform=tf, test=True)
    testloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    losses = train(net, 2, criterion, optimizer, trainloader, testloader=testloader, total_err=False)
    if report_errs:
        import matplotlib.pyplot as plt
        plt.plot([i for i in range(len(losses))], losses)
        plt.title("Test error over mini-batches, no noise")
        plt.xlabel("Mini-batch")
        plt.ylabel("Test Set Error")
        plt.savefig("simple_train_err.png")
    return net


def train_simple_mult(report_errs=False):
    net = train_simple()
    print("Train Last Layer:")
    tf = transforms.Compose([transforms.ToTensor()])#, transforms.Lambda(lambda x: x.cuda())])
    # tf = transforms.ToTensor()
    trainset = ImageLoaderMult(tform=tf)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    testset = ImageLoaderMult(tform=tf, test=True)
    testloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    criterion = nn.CrossEntropyLoss()
    params = list(net.named_parameters())
    optimizer = optim.SGD([params[-1][1], params[-2][1]], lr=0.001, momentum=0.9)

    losses = train(net, 2, criterion, optimizer, trainloader, testloader=testloader, total_err=report_errs, mult=True)
    if report_errs:
        import matplotlib.pyplot as plt
        plt.plot([i for i in range(len(losses))], losses)
        plt.title("Test error over mini-batches, no noise")
        plt.xlabel("Mini-batch")
        plt.ylabel("Test Set Error")
        plt.savefig("simple_train_err.png")
    return net

def train_uniform_noise(report_errs=False, var=.5):
    net = Net(3)
    net.cuda()
    net.zero_grad()
    tf = make_noisy_tf(var)
    trainset = ImageLoader(tform=tf)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    testset = ImageLoader(tform=tf, test=True)
    testloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    losses = train(net, 2, criterion, optimizer, trainloader, testloader=testloader, total_err=False)
    if report_errs:
        import matplotlib.pyplot as plt
        plt.plot([i for i in range(len(losses))], losses)
        plt.title("Test error over mini-batches, SNR = " + str(calc_var(trainset[0][0]) / var))
        plt.xlabel("Mini-batch")
        plt.ylabel("Test Set Error")
        plt.savefig("simple_train_err_noise_" + str(var) + ".png")
    return net

# net = Net(2)
# net.zero_grad()
#
# tf = make_noisy_tf(spread=1)
# trainset = ImageLoader(tform=tf)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                           shuffle=True, num_workers=2)
#
# # Should we still be doing pos / neg?
# testset = ImageLoader(tform=tf, test=True)
#
# testloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                           shuffle=True, num_workers=2)
#
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#
# train(net, 2, criterion, optimizer, trainloader, testloader=testloader)

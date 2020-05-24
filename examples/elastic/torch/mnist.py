# taken from https://github.com/pytorch/examples/blob/master/mnist/main.py

from __future__ import print_function

import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import runai.elastic.torch
import runai.ga.torch


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example with Run:AI Elasticity')
    parser.add_argument('--global-batch-size', type=int, default=256, metavar='N',
                        help='global batch size for training (default: 256)')
    parser.add_argument('--gpu-batch-size', type=int, default=32, metavar='N',
                        help='maximum GPU batch size for training (default: 32)')
    parser.add_argument('--samples', type=int, default=0, metavar='N',
                        help='number of samples to use from the train dataset (default: all)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--test-samples', type=int, default=0, metavar='N',
                        help='number of samples to use from the test dataset (default: all)')
    parser.add_argument('--shuffle', action='store_true', default=False,
                        help='shuffle datasets')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    return parser.parse_args()


def create_train_loader(args, data_dir, **kwargs):
    train_dataset = datasets.MNIST(data_dir, train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))

    if args.samples != 0:
        train_dataset.data = train_dataset.data[:args.samples]

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=runai.elastic.batch_size, shuffle=args.shuffle, **kwargs)

    return train_loader


def create_test_loader(args, data_dir, **kwargs):
    test_dataset = datasets.MNIST(data_dir, train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))

    if args.test_samples != 0:
        test_dataset.data = test_dataset.data[:args.test_samples]

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.test_batch_size, shuffle=args.shuffle, **kwargs)

    return test_loader


def create_optimizer(args, model):
    # randomally choose from the two options, for demo purposes
    if random.choice([True, False]):
        # create an optimizer
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)

        # wrap it with Run:AI GA
        optimizer = runai.ga.torch.optim.Optimizer(optimizer, runai.elastic.steps)
    else:
        # create a Run:AI GA optimizer
        optimizer = runai.ga.torch.optim.Adadelta(runai.elastic.steps, model.parameters(), lr=args.lr)

    return optimizer


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # arguments
    args = parse_args()
    torch.manual_seed(args.seed)

    # device arguments
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # init Run:AI elasticity
    runai.elastic.torch.init(args.global_batch_size, args.gpu_batch_size)

    # data directory
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    # load train data
    train_loader = create_train_loader(args, data_dir, **kwargs)

    # load test data
    test_loader = create_test_loader(args, data_dir, **kwargs)

    # create the network
    model = Net().to(device)

    # make the network data-parallelised
    model = nn.DataParallel(model)

    # create the optimizer
    optimizer = create_optimizer(args, model)

    # create the lr scheduler
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # run all epochs
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()


if __name__ == '__main__':
    main()

import argparse
import os
import random

import torch
from torchvision import datasets, models, transforms

import runai.elastic.torch
import runai.ga.torch


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training with Run:AI Elasticity')
    parser.add_argument('--model', default='resnet50', choices=['resnet50', 'resnet101', 'resnet152', 'vgg16', 'vgg19'],
                        help='neural network to use (default: resnet50)')
    parser.add_argument('--global-batch-size', type=int, default=256, metavar='N',
                        help='global batch size for training (default: 256)')
    parser.add_argument('--gpu-batch-size', type=int, default=64, metavar='N',
                        help='maximum GPU batch size for training (default: 32)')
    parser.add_argument('--samples', type=int, default=0, metavar='N',
                        help='number of samples to use from the train dataset (default: all)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--test-samples', type=int, default=0, metavar='N',
                        help='number of samples to use from the test dataset (default: all)')
    parser.add_argument('--shuffle', action='store_true', default=False,
                        help='shuffle datasets (default: False)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='learning rate')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    return parser.parse_args()


def create_train_loader(args, data_dir, **kwargs):
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]))

    if args.samples != 0:
        train_dataset.data = train_dataset.data[:args.samples]

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=runai.elastic.batch_size, shuffle=args.shuffle, **kwargs)

    return train_loader


def create_test_loader(args, data_dir, **kwargs):
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]))

    if args.test_samples != 0:
        test_dataset.data = test_dataset.data[:args.test_samples]

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=args.shuffle, **kwargs)

    return test_loader


def create_optimizer(args, model):
    kwargs = dict(
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4,
    )

    # randomally choose from the two options, for demo purposes
    if random.choice([True, False]):
        # create an optimizer
        optimizer = torch.optim.SGD(model.parameters(), **kwargs)

        # wrap it with Run:AI GA
        optimizer = runai.ga.torch.optim.Optimizer(optimizer, runai.elastic.steps)
    else:
        # create a Run:AI GA optimizer
        optimizer = runai.ga.torch.optim.SGD(runai.elastic.steps, model.parameters(), **kwargs)

    return optimizer


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    train_loss = 0
    total = 0
    correct = 0

    loss_fn = torch.nn.CrossEntropyLoss()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} | Acc: {:.3}% ({}/{})'.format(
                epoch,
                batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                train_loss / (batch_idx + 1),
                100. * correct / total, correct, total))


def test(model, device, test_loader):
    model.eval()

    test_loss = 0
    correct = 0

    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            test_loss += loss.item()
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()

    total = len(test_loader.dataset)
    test_loss /= total

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,
        correct, total, 100. * correct / total))


def main():
    # arguments
    args = parse_args()
    print("Using arguments: {}".format(args))

    # pick a device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}

    # init Run:AI elasticity
    runai.elastic.torch.init(args.global_batch_size, args.gpu_batch_size)

    # data directory
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    # load train data
    train_loader = create_train_loader(args, data_dir, **kwargs)

    # load test data
    test_loader = create_test_loader(args, data_dir, **kwargs)

    # create the network
    model = getattr(models, args.model)().to(device)

    # make the network data-parallelised
    model = torch.nn.DataParallel(model)

    # create the optimizer
    optimizer = create_optimizer(args, model)

    # run all epochs
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)


if __name__ == '__main__':
    main()

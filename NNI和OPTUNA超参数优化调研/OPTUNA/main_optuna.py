'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import logging

from models import *
from utils import progress_bar


_logger = logging.getLogger("cifar10_pytorch_automl")

trainloader = None
testloader = None
net = None
criterion = None
optimizer = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0.0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# Training
def train(epoch, batches=-1):
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        acc = 100.*correct/total

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        if batches > 0 and (batch_idx+1) >= batches:
            return

def addtest(epoch):
    global best_acc
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            acc = 100.*correct/total

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc
    return acc, best_acc

def objective(trial):
    args = {}
    lr = trial.suggest_float("lr", 0.00001, 0.1, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "Adadelta", "Adagrad", "Adam", "Adamax"])
    model = trial.suggest_categorical("model", ["vgg", "resnet18", "googlenet", "densenet121", "mobilenet", "dpn92", "senet18"])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    args['lr'] = lr
    args['optimizer'] = optimizer_name
    args['model'] = model
    args['batch_size'] = batch_size
    args['epochs'] = 100
    def prepare(args):

        global trainloader
        global testloader
        global net
        global criterion
        global optimizer

        # Data
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args['batch_size'], shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args['batch_size'], shuffle=False, num_workers=2)

        # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        # Model
        print('==> Building model..')
        if args['model'] == 'vgg':
            net = VGG('VGG19')
        if args['model'] == 'resnet18':
            net = ResNet18()
        if args['model'] == 'googlenet':
            net = GoogLeNet()
        if args['model'] == 'densenet121':
            net = DenseNet121()
        if args['model'] == 'mobilenet':
            net = MobileNet()
        if args['model'] == 'dpn92':
            net = DPN92()
        if args['model'] == 'shufflenetg2':
            net = ShuffleNetG2()
        if args['model'] == 'senet18':
            net = SENet18()

        net = net.to(device)
        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(net.parameters(), lr=args['lr'], momentum=0.9, weight_decay=5e-4)

        if args['optimizer'] == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=args['lr'], momentum=0.9, weight_decay=5e-4)
        if args['optimizer'] == 'Adadelta':
            optimizer = optim.Adadelta(net.parameters(), lr=args['lr'])
        if args['optimizer'] == 'Adagrad':
            optimizer = optim.Adagrad(net.parameters(), lr=args['lr'])
        if args['optimizer'] == 'Adam':
            optimizer = optim.Adam(net.parameters(), lr=args['lr'])
        if args['optimizer'] == 'Adamax':
            optimizer = optim.Adam(net.parameters(), lr=args['lr'])

    prepare(args)
    for epoch in range(start_epoch, start_epoch + args['epochs']):
        train(epoch,  -1)
        acc, best_acc = addtest(epoch)
    return best_acc

if __name__ == '__main__':
    import optuna
    from optuna.trial import TrialState


    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=100000)

    pruned_trials = study.get_trials(deepcopy=False, states=(TrialState.PRUNED,))
    complete_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print("Best trial:")
    trial = study.best_trial
    print("Value: ", trial.value)
    print("Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
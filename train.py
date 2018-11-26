import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from collections import OrderedDict
from torchvision import datasets, transforms, models

def train(data_dir, arch_name, learning_rate, hidden_units, epochs, gpu, save_dir, labels_count):
    device = torch.device('cuda' if torch.cuda.is_available() and gpu else "cpu")
    train_data, test_data = get_data_and_loaders(data_dir)

    # Setup the loaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=4)

    model, criterion, optimizer = setup_network(arch_name, learning_rate, labels_count)

    # Taken from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    train_network(model, criterion, optimizer, trainloader, testloader,
                  len(train_data), len(train_data), scheduler, epochs, device)
    save_model_checkpoint(arch_name, model, train_data, hidden_units, labels_count, save_dir)

def getArch(arch_name):
    archs = {
        'vgg13': {
            'model': models.vgg13(pretrained=True),
            'classifier_inputs': 25088,
            'hidden_layer_outs': 4096
        },
        'vgg16': {
            'model': models.vgg16(pretrained=True),
            'classifier_inputs': 25088,
            'hidden_layer_outs': 4096
        },
        'vgg19': {
            'model': models.vgg19(pretrained=True),
            'classifier_inputs': 25088,
            'hidden_layer_outs': 4096
        },
        'densenet121': {
            'model': models.densenet121(pretrained=True),
            'classifier_inputs': 1024,
            'hidden_layer_outs': 120
        },
        'alexnet': {
            'model': models.alexnet(pretrained=True),
            'classifier_inputs': 9216,
            'hidden_layer_outs': 120
        }
    }

    return archs[arch_name]

def get_data_and_loaders(data_dir):
    train_dir = data_dir + '/train'
    test_dir = data_dir + '/test'

    image_norm_mean = [0.485, 0.456, 0.406]
    image_norm_std = [0.229, 0.224, 0.225]
    train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomRotation(30),
                                           transforms.ToTensor(),
                                           transforms.Normalize(image_norm_mean,
                                                                image_norm_std)])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(image_norm_mean,
                                                               image_norm_std)])

    # Step: Data loading
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    return train_data, test_data

def setup_network(arch_name, learning_rate, labels_count):
    arch = getArch(arch_name)
    model = arch['model']

    # Step: Pretrained Network
    # Freeze the params
    for param in model.parameters():
        param.requires_grad = False

    # Step: Feedforward Classifier
    classifier = nn.Sequential(OrderedDict([('dropout', nn.Dropout(p=0.15)),
                                            ('fc1', nn.Linear(arch['classifier_inputs'],
                                                              arch['hidden_layer_outs'])),
                                            ('relu1', nn.ReLU()),
                                            ('fc2', nn.Linear(arch['hidden_layer_outs'],
                                                              labels_count)),
                                            ('relu2', nn.ReLU()),
                                            ('output', nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    return model, criterion, optimizer

def train_network(model,
                  criterion,
                  optimizer,
                  trainloader,
                  testloader,
                  traindata_len,
                  testdata_len,
                  scheduler,
                  epochs,
                  device='cpu'):

    model.to(device)

    # Step: Training the network
    for epoch in range(epochs):
        print('Starting epochs[{}]'.format(epoch))

        # Train
        running_loss = 0.0
        running_corrects = 0
        scheduler.step()
        model.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model.forward(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Stats
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / traindata_len
        epoch_acc = running_corrects.double() / traindata_len
        print('Training loss epochs[{}]: {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))

        # Validate
        # running_loss = 0.0
        # running_corrects = 0
        # model.eval()
        # for inputs, labels in testloader:
        #     inputs, labels = inputs.to(device), labels.to(device)
        #     optimizer.zero_grad()

        #     with torch.set_grad_enabled(False):
        #         outputs = model(inputs)
        #         _, preds = torch.max(outputs, 1)
        #         loss = criterion(outputs, labels)

        #     # Stats
        #     running_loss += loss.item() * inputs.size(0)
        #     running_corrects += torch.sum(preds == labels.data)

        # epoch_loss = running_loss / testdata_len
        # epoch_acc = running_corrects.double() / testdata_len
        # print('Validation loss epochs[{}]: {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))


def save_model_checkpoint(arch_name, model, train_data, hidden_units, labels_count, save_dir):
    # Directories are a pain to deal with, so I'm leaving them for now as it doesn't really prove anything
    # os.mkdir(save_dir)
    arch = getArch(arch_name)
    model.class_to_idx = train_data.class_to_idx
    model.cpu
    torch.save({'arch_name': arch_name,
                'hidden_units': hidden_units,
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx,
                'input_size': arch['classifier_inputs'],
                'output_size': labels_count},
                '{0}_checkpoint_{1}_{2}.pth'.format(save_dir, arch_name, hidden_units))

# CMD
parser = argparse.ArgumentParser()
parser.add_argument('data_dir',
                    type=str,
                    help='Where\'s your data?',
                    default='flowers')
parser.add_argument('--arch',
                    type=str,
                    help='Which pretrained architechture to use',
                    default='vgg13')
parser.add_argument('--learning_rate',
                    type=float,
                    help='The rate at which the training will learn',
                    default=0.001)
parser.add_argument('--hidden_units',
                    type=int, help='How many hidden layers in the classifier training',
                    default=512)
parser.add_argument('--epochs',
                    type=int,
                    help='How many rounds we\'ll do the forwardpass and back propegation',
                    default=24)
parser.add_argument('--gpu',
                    help='To GPU, or not to GPU? (to GPU)',
                    action='store_true',
                    default=True)
parser.add_argument('--save_dir',
                    help='Where we will keep you checkpoints safe for you',
                    type=str,
                    default='checkpoints')
parser.add_argument('--labels_count',
                    help='How many classifications are you going to have?',
                    type=int,
                    default=102)

args, _ = parser.parse_known_args()

# we could move all the functions out into a helper file as we share
# `setup_network` with predict, but this just keeps it together.
if __name__ == '__main__':
  train(
      args.data_dir,
      args.arch,
      args.learning_rate,
      args.hidden_units,
      args.epochs,
      args.gpu,
      args.save_dir,
      args.labels_count)

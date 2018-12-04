import argparse
import json
import os
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

from workspace_utils import keep_awake

def train(data_dir, arch_name, learning_rate, hidden_units, epochs, gpu, save_dir, num_labels):
    device = torch.device('cuda' if torch.cuda.is_available() and gpu else "cpu")
    train_data, test_data, valid_data = get_data_and_loaders(data_dir)

    # Setup the loaders
    model = load_model(arch_name, learning_rate, hidden_units, num_labels)
    train_network(model,
                  train_data,
                  valid_data,
                  learning_rate,
                  epochs,
                  device)

    save_model_checkpoint(arch_name, model, train_data, hidden_units, learning_rate)

def getArch(arch_name):
    print('Looking for {}'.format(arch_name))
    if arch_name == 'vgg13':
        return {
            'model': models.vgg13(pretrained=True),
            'classifier_inputs': 25088,
            'hidden_layer_outs': 4096
        }

    if arch_name == 'vgg16':
        return  {
            'model': models.vgg16(pretrained=True),
            'classifier_inputs': 25088,
            'hidden_layer_outs': 4096
        }

    if arch_name == 'vgg19':
        return {
            'model': models.vgg19(pretrained=True),
            'classifier_inputs': 25088,
            'hidden_layer_outs': 4096
        }

    if arch_name == 'densenet121':
        return {
            'model': models.densenet121(pretrained=True),
            'classifier_inputs': 1024,
            'hidden_layer_outs': 120
        }

    if arch_name == 'alexnet':
        return {
            'model': models.alexnet(pretrained=True),
            'classifier_inputs': 9216,
            'hidden_layer_outs': 120
        }

def get_data_and_loaders(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
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
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(image_norm_mean,
                                                                image_norm_std)])

    # Step: Data loading
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    return train_data, test_data, valid_data

def load_model(arch_name, learning_rate, hidden_units, num_labels):
    arch = getArch(arch_name)
    model = arch['model']
    dropout = 0.5

    num_filters = arch['classifier_inputs']

    # Step: Pretrained Network
    # Freeze the params
    for param in model.parameters():
        param.requires_grad = False

    # Step: Feedforward Classifier
    classifier = nn.Sequential(OrderedDict([
            ('dropout1', nn.Dropout(dropout)),
            ('fc1', nn.Linear(num_filters, hidden_units)),
            ('relu1', nn.ReLU(True)),
            ('dropout2', nn.Dropout(dropout)),
            ('fc2', nn.Linear(hidden_units, hidden_units)),
            ('relu2', nn.ReLU(True)),
            ('fc3', nn.Linear(hidden_units, num_labels)),
            ]))

    model.classifier = classifier
    return model

def train_network(model,
                  train_data,
                  valid_data,
                  learning_rate,
                  epochs,
                  device='cpu'):

    dataloaders = {
        'train': torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True),
        'valid': torch.utils.data.DataLoader(valid_data, batch_size=4)
    }
    data_sizes = {
        'train': len(train_data),
        'valid': len(valid_data)
    }

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    # Optimizing only the params that need optimisation
    optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=learning_rate, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    # I found this method gave me the best results, and seemed a little more
    # complete and logical
    for epoch in keep_awake(range(epochs)):
        print('Starting Epoch {}/{}'.format(epoch + 1, epochs))
        print('~' * 12)

        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / data_sizes[phase]
            epoch_acc = running_corrects.double() / data_sizes[phase]

            print('{} Loss: {:.4f} Accuracy: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Store class_to_idx into a model property
    model.class_to_idx = train_data.class_to_idx
    return model


def save_model_checkpoint(arch_name, model, train_data, hidden_units, learning_rate):
    model.cpu
    model.class_to_idx = train_data.class_to_idx
    torch.save({'structure': arch_name,
                'hidden_layer1': 120,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                'checkpoint_{}_{}_{}.pth'.format(arch_name, hidden_units, learning_rate))

# CMD
parser = argparse.ArgumentParser()
parser.add_argument('data_dir',
                    type=str,
                    help='Where\'s your data?',
                    default='flowers')
parser.add_argument('--arch',
                    type=str,
                    help='Which pretrained architechture to use',
                    default='vgg19')
parser.add_argument('--learning_rate',
                    type=float,
                    help='The rate at which the training will learn',
                    default=0.001)
parser.add_argument('--hidden_units',
                    type=int, help='How many hidden layers in the classifier training',
                    default=4096)
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
parser.add_argument('--num_labels',
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
      args.num_labels)

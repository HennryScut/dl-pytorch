import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True  # 使用确定性算法，避免不确定性导致结果无法重现
##########################
### SETTINGS
##########################
def init(batch_size):
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ##########################
    ### MNIST DATASET
    ##########################

    # Note transforms.ToTensor() scales input images
    # to 0-1 range
    train_dataset = datasets.MNIST(root='../data',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

    valid_dataset = torch.utils.data.Subset
    test_dataset = datasets.MNIST(root='../data',
                                  train=False,
                                  transform=transforms.ToTensor(),
                                  download=True)


    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    # Checking the dataset
    for images, labels in train_loader:
        print('Image batch dimensions:', images.shape)
        print('Image label dimensions:', labels.shape)
        break

    return train_loader, test_loader, device

##########################
### MODEL
##########################


class ConvNet(torch.nn.Module):

    def __init__(self, num_classes):
        super(ConvNet, self).__init__()

        # calculate same padding:
        # (w - k + 2*p)/s + 1 = o
        # => p = (s(o-1) - w + k)/2

        # 28x28x1 => 28x28x4
        self.conv_1 = torch.nn.Conv2d(in_channels=1,
                                      out_channels=4,
                                      kernel_size=(3, 3),
                                      stride=(1, 1),
                                      padding=1)  # (1(28-1) - 28 + 3) / 2 = 1
        # 28x28x4 => 14x14x4
        self.pool_1 = torch.nn.MaxPool2d(kernel_size=(2, 2),
                                         stride=(2, 2),
                                         padding=0)  # (2(14-1) - 28 + 2) = 0
        # 14x14x4 => 14x14x8
        self.conv_2 = torch.nn.Conv2d(in_channels=4,
                                      out_channels=8,
                                      kernel_size=(3, 3),
                                      stride=(1, 1),
                                      padding=1)  # (1(14-1) - 14 + 3) / 2 = 1
        # 14x14x8 => 7x7x8
        self.pool_2 = torch.nn.MaxPool2d(kernel_size=(2, 2),
                                         stride=(2, 2),
                                         padding=0)  # (2(7-1) - 14 + 2) = 0

        self.linear_1 = torch.nn.Linear(7 * 7 * 8, num_classes)

        ###############################################
        # Reinitialize weights using He initialization
        ###############################################
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.detach())
                m.bias.detach().zero_()
            elif isinstance(m, torch.nn.Linear):
                nn.init.kaiming_normal_(m.weight.detach())
                m.bias.detach().zero_()

    def forward(self, x):
        out = self.conv_1(x)
        out = F.relu(out)
        out = self.pool_1(out)

        out = self.conv_2(out)
        out = F.relu(out)
        out = self.pool_2(out)

        logits = self.linear_1(out.view(-1, 7 * 7 * 8))
        probas = F.softmax(logits, dim=1)
        return logits, probas


def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    for features, targets in data_loader:
        features = features.to(device)
        targets = targets.to(device)
        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100


def train(Model, trainloader, testloader, num_epochs):
    start_time = time.time()
    for epoch in range(num_epochs):
        Model = Model.train()
        for batch_idx, (features, targets) in enumerate(trainloader):

            features = features.to(device)
            targets = targets.to(device)

            ### FORWARD AND BACK PROP
            logits, probas = Model(features)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()

            loss.backward()

            ### UPDATE MODEL PARAMETERS
            optimizer.step()

            ### LOGGING
            if not batch_idx % 50:
                print('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f'
                      % (epoch + 1, num_epochs, batch_idx,
                         len(train_loader), loss))

        Model = Model.eval()
        print('Epoch: %03d/%03d training accuracy: %.2f%%' % (
            epoch + 1, num_epochs,
            compute_accuracy(Model, trainloader)))

        print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))

    print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))
    print('Test accuracy: %.2f%%' % (compute_accuracy(Model, testloader)))


def save(Model, optimizer, path):
    # save model
    states = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
    torch.save(states, path)


if __name__=='__main__':
    # Hyperparameters
    random_seed = 1
    learning_rate = 0.05
    num_epochs = 5
    batch_size = 128
    # Architecture
    num_classes = 10
    # init
    train_loader, test_loader, device = init(batch_size)
    # model
    torch.manual_seed(random_seed)
    model = ConvNet(num_classes=num_classes)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    #train
    train(model, train_loader, test_loader, num_epochs)
    # save
    PATH = 'cnn_Hinit_params.tar'
    save(model, optimizer, PATH)
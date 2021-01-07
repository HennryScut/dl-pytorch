import torch
import torch.nn as nn
import torch.nn.functional as F
from unlinear.dataset import TrainSet
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(1, 20)
        self.linear2 = nn.Linear(20, 40)
        self.linear3 = nn.Linear(40, 1)

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        return x

writer = SummaryWriter('logs')


def trainfunc(model, optimizer, epoch, train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        y_pred = model(data)
        loss = F.mse_loss(y_pred, target, reduction='mean')
        loss.backward()
        optimizer.step()
        writer.add_scalar(str(epoch)+'loss', loss.item(), batch_idx)
        if (batch_idx+1) % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


writer.close()
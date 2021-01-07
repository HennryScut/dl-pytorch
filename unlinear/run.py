import torch
from unlinear.dataset import TrainSet, TestSet
from torch.utils.data import DataLoader
from unlinear.model import Model, trainfunc


traindata = TrainSet()
testdata = TestSet()
print(traindata[0])
trainloader = DataLoader(traindata, batch_size=50, shuffle=True)
testloader = DataLoader(testdata, batch_size=50, shuffle=True)

model = Model()
optim = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.1)

for epoch in torch.arange(20):
    trainfunc(model, optim, epoch, trainloader)
    # for batch_idx, (x, y) in enumerate(trainloader):
        # if batch_idx == 0:
        #     print('epoch:', epoch, '\t')
        #     print('input:', x[0], '\t')
        #     print('input:', y[0], '\n')


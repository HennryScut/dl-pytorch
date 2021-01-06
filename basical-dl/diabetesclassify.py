import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset('/data/diabetes.csv.gz')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.Linear1 = torch.nn.Linear(8, 6)
        self.Linear2 = torch.nn.Linear(6, 4)
        self.Linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.Linear1(x))
        x = self.sigmoid(self.Linear2(x))
        x = self.sigmoid(self.Linear3(x))
        return x


model = Model()
# GPU计算
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model.to(device)

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

if __name__=='__main__':
    for epoch in range(20):
        for i, data in enumerate(train_loader, 0):
            inputs, label = data

            y_pred = model(inputs)

            loss = criterion(y_pred, label)
            print(epoch, i, loss.item())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

count = 0
for x, y in dataset:
    y_p = model(x)
    if y_p>0.5:
        y_p = 1
    else:
        y_p = 0

    if y_p == y:
        count += 1

rate = count/dataset.len
print('accuracy', rate)

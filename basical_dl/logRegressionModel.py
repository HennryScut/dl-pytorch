import torch

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[0.0], [0.0], [1.0]])


class logRegressionModel(torch.nn.Module):
    def __init__(self):
        super(logRegressionModel, self).__init__()
        self.Linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.Linear(x))
        return y_pred


model = logRegressionModel()

criterion = torch.nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.data.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# output weight and bias
print('w=', model.Linear.weight.item())
print('b=', model.Linear.bias.item())

# test model
x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print('y_pred=', y_test.data.item())
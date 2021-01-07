import torch
from torch.utils.data import Dataset

# train value
# y = x**2 + 6.8
true_x = torch.arange(-100, 100, step=0.1).view(-1, 1)
true_y = true_x**2 + 6.8


class TrainSet(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        return true_x[index], true_y[index]

    def __len__(self):
        return len(true_x)


# test value
# y = x**2 + 6.8
test_x = torch.arange(-100, 100, step=0.1).view(-1, 1)
test_y = test_x**2 + 6.8


class TestSet(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        return test_x[index], test_y[index]

    def __len__(self):
        return len(test_x)
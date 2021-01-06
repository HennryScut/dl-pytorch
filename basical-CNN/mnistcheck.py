import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter

# 设定超参数
BATCH_SIZE=512 # 批次大小
EPOCHS=20 # 总共训练批次
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多

# 创建tensorboard日志
Writer_train = SummaryWriter('logs/train')
Writer_test = SummaryWriter('logs/test')

# 下载并读取数据集
train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)


test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)


# 经典卷积神经网络
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入图像，通道1，尺寸28x28
        # 卷积池化层
        self.conv1=nn.Conv2d(1, 10, 5)  # 输入通道1，输出通道（卷积核数）10，输出图像尺寸24x24
        self.pool = nn.MaxPool2d(2, 2)  # 最大池化，输出图像尺寸12x12
        self.conv2=nn.Conv2d(10, 20, 3)  # 输入通道10，输出通道20，输出图像尺寸10×10
        # 全连接层
        self.fc1 = nn.Linear(20*10*10, 500)  # 输入尺寸：通道数*输入图像尺寸，20*10*10；输出尺寸500
        self.fc2 = nn.Linear(500, 10)  # 输入尺寸500， 输出尺寸10，对应10分类

    def forward(self, x):
        # (batch_size，channels，x，y) = tensor.size()
        in_size = x.size(0)  # 获取输入batch数据的样本数量，即batch_size
        out = self.conv1(x)  # 24
        out = F.relu(out)
        out = self.pool(out)   # 12
        out = self.conv2(out)  # 10
        out = F.relu(out)
        out = out.view(in_size, -1)  # 将卷积层输入展开为batch_size个行向量，作为全连接层输入
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)  # 对输出每行进行归一化计算
        return out


# googleNet inspection
class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1_1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch5_5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5_5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3_3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3_3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3_3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_1_1_pool = nn.Conv2d(in_channels, 24, kernel_size=1)
        self.branch_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        branch1_1 = self.branch1_1(x)

        branch5_5 = self.branch5_5_1(x)
        branch5_5 = self.branch5_5_2(branch5_5)

        branch3_3 = self.branch3_3_1(x)
        branch3_3 = self.branch3_3_2(branch3_3)
        branch3_3 = self.branch3_3_3(branch3_3)

        branch_pool = self.branch_pool(x)
        branch_pool = self.branch_1_1_pool(branch_pool)

        outputs = [branch1_1, branch5_5, branch3_3, branch_pool]
        return torch.cat(outputs, dim=1)


class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)

        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)

        self.mp = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(1408, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


# ResNet
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        y = F.relu(x+y)
        return y


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.mp = nn.MaxPool2d(kernel_size=2)

        self.resblock1 = ResidualBlock(16)
        self.resblock2 = ResidualBlock(32)

        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.resblock1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.resblock2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


# 训练
def train(model, device, train_loader, optimizer, epoch):  # 定义每个epoch的训练细节
    model.train()  # 设置为training模式
    for batch_idx, (data, target) in enumerate(train_loader):  # 读入1 batch数据
        data, target = data.to(device), target.to(device)  # 数据写入GPU
        optimizer.zero_grad()  # 优化器的梯度值初始化为0
        output = model(data)  # 前向传播
        loss = F.nll_loss(output, target)  # 计算negative log likelihood loss损失函数
        loss.backward()  # 反向传播
        optimizer.step()  # 结束一次前传+反传之后，更新优化器参数
        if (batch_idx + 1) % 30 == 0:  # 每30个batch输出一次
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    Writer_train.add_scalar('loss', loss.item(), epoch)


# 测试
def test(model, device, test_loader, epoch):
    model.eval()  # 设置为test模式
    test_loss = 0  # 初始化测试损失值为0
    correct = 0  # 初始化预测正确的数据个数为0
    with torch.no_grad():  # 以下关闭自动梯度计算，降低内存消耗
        for data, target in test_loader:  # 读入1 batch数据
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # 将一批batch的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到每一行（样本）概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()  # 对预测正确的数据个数进行累加

    test_loss /= len(test_loader.dataset)  # 因为把所有loss值进行过累加，所以最后要除以总得数据长度才得平均loss
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    Writer_test.add_scalar('loss', test_loss, epoch)

# 运行接口
if __name__=='__main__':
    # for epoch in range(1, EPOCHS + 1):  # 训练/测试周期循环
    #     # 经典卷积网络
    #     model = ConvNet().to(DEVICE)  # 实例化网络模型并转入GPU
    #     optimizer = optim.Adam(model.parameters())
    #
    #     train(model, DEVICE, train_loader, optimizer, epoch)
    #     test(model, DEVICE, test_loader)
    #
    # for epoch in range(1, EPOCHS + 1):  # 训练/测试周期循环
    #     # GoogleNet
    #     model = GoogleNet().to(DEVICE)  # 实例化网络模型并转入GPU
    #     optimizer = optim.Adam(model.parameters())
    #
    #     train(model, DEVICE, train_loader, optimizer, epoch)
    #     test(model, DEVICE, test_loader)

    for epoch in range(1, EPOCHS + 1):  # 训练/测试周期循环
        # ResNet网络
        model = ResNet().to(DEVICE)  # 实例化网络模型并转入GPU
        optimizer = optim.Adam(model.parameters())

        train(model, DEVICE, train_loader, optimizer, epoch)
        test(model, DEVICE, test_loader, epoch)

    Writer_train.close()
    Writer_test.close()
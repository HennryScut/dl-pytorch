import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from basical_CNN.cnn_Hinit import ConvNet

batch_size = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


test_dataset = datasets.MNIST(root='../data',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)

PATH = 'cnn_Hinit_params.tar'
model = ConvNet(10)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.01)
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)  # 由于是在GPU下保存，因此也要继续移植到GPU
model.eval()  # 设定为评估模式
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

writer = SummaryWriter('logs/save_load')

right_num = torch.tensor(0)
writer_num = 0
for idx, (input, target) in enumerate(test_loader):
    input = input.to(device)  # 模型在GPU上，数据也得迁移到GPU，并且不支持就地处理，需要重新赋值
    target = target.to(device)
    _, pred_y = model(input)
    _, max_index = torch.max(pred_y, dim=1)
    if idx % 1000 == 0:
        writer.add_image(str(max_index.item()), input[0], writer_num)
        print('pred_index:', max_index.item())  # 预测的结果
        print('true_class:', target.data.item(), '\n')  # 实际结果
        # print('pred_tensor:', pred_y, '\t')  # 预测概率向量
        writer_num += 1
    if max_index == target:
        right_num += 1
print('accuracy:', right_num.float()/len(test_loader)*100)

writer.close()
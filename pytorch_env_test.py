import torch

print('torch version:', torch.__version__)
print('CUDA GPU check', torch.cuda.is_available())
if(torch.cuda.is_available()):
    print('CUDA GPU num:',torch.cuda.device_count())
    n = torch.cuda.device_count()
while n>0:
    print('CUDA GPU name:', torch.cuda.get_device_name())
    n-=1
print('CUDA GPU index:',torch.cuda.current_device())

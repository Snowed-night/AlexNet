import torch
from torch import nn
from torch.nn import AvgPool2d
from torchsummary import summary     #设置模型参数细节

class LeNet(nn.Module):
    # 参数初始化
    def __init__(self):
        super(LeNet,self).__init__()
        #卷积+激活+池化
        self.c1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,padding=2)
        self.sig=nn.Sigmoid()
        self.s2=nn.AvgPool2d(kernel_size=2,stride=2)
        self.c3=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        self.s4=nn.AvgPool2d(kernel_size=2,stride=2)
        #平展
        self.flatten=nn.Flatten()
        self.f5 = nn.Linear(in_features=400,out_features=120)
        self.f6 = nn.Linear(in_features=120, out_features=84)
        self.f7 = nn.Linear(in_features=84, out_features=10)
    #前向传播
    def forward(self, x):
        #卷积+激活+池化
        x=self.sig(self.c1(x))
        x=self.s2(x)
        x=self.sig(self.c3(x))
        x=self.s4(x)
        x=self.flatten(x)
        #平展
        x=self.f5(x)
        x=self.f6(x)
        x=self.f7(x)
        return x

if __name__ == '__main__':
    #判断是否有cpu
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #模型实地化
    model = LeNet().to(device)
    print(summary(model, (1,28,28)))




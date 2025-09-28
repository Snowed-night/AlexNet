import copy
import time
import pandas as pd
import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as data
import pandas
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
#导入网络模型
from model import LeNet    #要改的地方1：模型名称

#处理训练集和验证集
def train_val_data_process():
    #加载数据
    train_data = FashionMNIST(root='./data',
                             train=True,
                             transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),      #要改的地方2：size设置为图片大小28*28
                             download=True)
    #划分数据
    train_data,val_data=data.random_split(train_data,[round(0.8*len(train_data)),round(0.2*len(train_data))])
    #每128个数据划分为一批次并打乱
    train_dataloader=data.DataLoader(dataset=train_data,        #要改的地方3：根据设备不同显存选择同时处理的一批次数量
                                    batch_size=128,
                                    shuffle=True,
                                    num_workers=8)

    val_dataloader = data.DataLoader(dataset=val_data,
                                      batch_size=128,
                                      shuffle=False,
                                      num_workers=8)

    return train_dataloader,val_dataloader

#训练模型函数
def train_model_process(model,train_dataloader,val_dataloader,num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #梯度下降Adam优化器模型的参数以最小化损失函数
    #model.parameters()：获取模型中的所有可训练参数（即需要被优化的权重和偏置）
    #lr=0.001：设置学习率为0.001，控制每次参数更新的步长。
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    #分类用交叉熵损失函数(回归用均方)
    criterion = nn.CrossEntropyLoss()
    #将模型放入设备
    model=model.to(device)
    #保存最好模型
    best_model_wts = copy.deepcopy(model.state_dict())
    #初始化参数
    #最高准确度
    best_acc = 0.0
    # 训练集损失列表
    train_loss_all = []
    # 验证集损失列表
    val_loss_all = []
    # 训练集准确度列表
    train_acc_all = []
    # 验证集准确度列表
    val_acc_all = []
    #保存时间
    since = time.time()
    #反向传播训练次数
    for epoch in range(num_epochs):
        #打印当前训练轮次
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        print("-" * 10)
        #初始化参数((训练和验证集的)损失+准确度+样本数量)
        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0
        train_num=0
        val_num=0
        #遍历每一批次，对每一个batch训练和计算
        for step,(b_x,b_y)in enumerate(train_dataloader):
            #将特征和标签放入训练设备
            b_x=b_x.to(device)
            b_y=b_y.to(device)
            #设置为训练模式
            model.train()
            #前向传播得到一个batch的预测输出（向量形式）
            output=model(b_x)
            #找出每个样本预测概率最高的类别作为预测标签
            pre_lab=torch.argmax(output,dim=1)
            #计算损失函数
            loss = criterion(output,b_y)
            #清除之前计算的梯度，防止梯度累积
            optimizer.zero_grad()
            #反向传播计算损失相对于模型参数的梯度
            loss.backward()
            #更新参数，以降低loss
            optimizer.step()
            #累加损失函数(总loss+=一批次loss=平均值loss*样本数量)
            train_loss += loss.item()*b_x.size(0)
            #预测正确，准确度加1,数量加1(算准确率)
            train_acc += torch.sum(pre_lab==b_y.data)
            train_num += b_x.size(0)

        #验证每一批次
        for step, (b_x, b_y) in enumerate(val_dataloader):
            # 将特征和标签放入验证设备
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            #设置验证模式
            model.eval()
            output=model(b_x)
            pre_lab=torch.argmax(output,dim=1)
            loss = criterion(output,b_y)
            val_loss += loss.item()*b_x.size(0)
            val_acc += torch.sum(pre_lab==b_y.data)
            val_num += b_x.size(0)

        #将该一批次训练集和验证集的平均loss值和精确率加入数组
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_acc.double().item() / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_acc.double().item() / val_num)
        #打印该轮次
        print("{}Train Loss:{:.4f} Train Acc{:.4f}".format(epoch,train_loss_all[-1],train_acc_all[-1]))
        print("{}Val Loss:{:.4f} Val Acc{:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))

        #更新最优准确度和模型
        if val_acc_all[-1]>best_acc:
            best_acc=val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        #计算用时
        time_use = time.time() - since
        print("训练用时{:.0f}m{:.0f}s".format(time_use//60, time_use%60))
    #保存最优参数和模型
    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts,"D:/pycharm projects/LeNet5/best_model.pth")        #要改的地方4：保存模型路径

    #将值做成表格
    train_process=pd.DataFrame(data={"epoch":range(num_epochs),
                                     "train_loss_all":train_loss_all,
                                     "val_loss_all": val_loss_all,
                                     "train_acc_all":train_acc_all,
                                     "val_acc_all":val_acc_all})
    return train_process

#画图
def matplot_acc_loss(train_process):
    plt.figure(figsize=(12,4))
    #一行两列第一张图
    plt.subplot(1,2,1)
    plt.plot(train_process["epoch"],train_process.train_loss_all,'ro-',label='train loss')  #红色
    plt.plot(train_process["epoch"],train_process.val_loss_all, 'bs-', label='val loss')  #蓝色
    plt.legend()
    plt.xlabel('epoch')  #x轴是轮次
    plt.ylabel('loss')
    #第二张图
    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"],train_process.train_acc_all, 'ro-', label='train acc')
    plt.plot(train_process["epoch"],train_process.val_acc_all, 'bs-', label='val acc')
    plt.legend()
    plt.xlabel('epoch')  # x轴是轮次
    plt.ylabel('acc')
    plt.legend()
    plt.show()

#主函数
if __name__ == '__main__':
    #加载模型
    LeNet1=LeNet()       #要改的地方5：加载模型
    train_dataloader,val_dataloader=train_val_data_process()
    train_process = train_model_process(LeNet1,train_dataloader,val_dataloader,100)  #要改的地方6:训练轮数
    matplot_acc_loss(train_process)
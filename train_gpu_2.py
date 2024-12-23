import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils import tensorboard
from torch.utils.tensorboard import SummaryWriter

device=torch.device("cuda")
print(device)
train_data=torchvision.datasets.CIFAR10(root='./data', train=True, download=False,transform=torchvision.transforms.ToTensor())
test_data=torchvision.datasets.CIFAR10(root='./data', train=False, download=False,transform=torchvision.transforms.ToTensor())
train_data_size= len(train_data)
test_data_size=len(test_data)
print("train size == {} ".format(train_data_size) )
print("test size == {} ".format(test_data_size) )
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        self.model =  nn.Sequential(
            nn.Conv2d(3, 32, 5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
        )
    def forward(self,x):
        x=self.model(x)
        return x

tudui = Tudui()
tudui.to(device)
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)


learning_rate = 0.01
optimizer =  torch.optim.SGD(tudui.parameters(), lr=learning_rate)

total_train_step  = 0
total_test_step=0
epoch = 10
writer=SummaryWriter("../logs_train")
import time
start_time=time.time()
for i in range(epoch):
    print("----------the {}th epoch--------".format(i+1))
    #train
    tudui.train()
    for data in train_dataloader:
        imgs,targets = data
        imgs=imgs.to(device)
        targets=targets.to(device)
        outputs = tudui(imgs)
        loss=0
        loss=loss_fn(outputs,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if(total_train_step % 100 == 0):
            end_time = time.time()
            print(end_time - start_time)
            start_time=time.time()
            print("the {}th train, loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar('train_loss',loss.item(),total_train_step)

    #test
    tudui.eval()
    total_test_loss=0
    total_accuracy=0
    with torch.no_grad():
        for data in test_dataloader:
            imgs , targets = data
            imgs=imgs.to(device)
            targets=targets.to(device)
            outputs=tudui(imgs)
            loss=loss_fn(outputs,targets)
            total_test_loss += loss.item()
            accuracy=(outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    #show
    print("total loss in test set : {}".format(total_test_loss))
    print("total accuracy in test set : {}".format(total_accuracy))
    writer.add_scalar('test_loss',total_test_loss,total_test_step)
    writer.add_scalar('test_accuracy',total_accuracy/test_data_size,total_test_step)
    total_test_step+=1

    #save
    torch.save(tudui,"tudui_{}.pth".format(i))
    print("save the model")
writer.close()

#网络模型，图像，标注，损失函数

from torch.utils import tensorboard
from torch.utils.tensorboard import SummaryWriter
from load_datasets import *
from model import *

train_data_size,test_data_size,train_dataloader,test_dataloader=load_data()
tudui = Tudui()
loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer =  torch.optim.SGD(tudui.parameters(), lr=learning_rate)
total_train_step  = 0
total_test_step=0
epoch = 10
writer=SummaryWriter("../logs_train")

for i in range(epoch):
    print("----------the {}th epoch--------".format(i+1))


    #train
    tudui.train()
    for data in train_dataloader:
        imgs,targets = data
        outputs = tudui(imgs)
        loss=0
        loss=loss_fn(outputs,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if(total_train_step % 100 == 0):
            print("the {}th train, loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar('train_loss',loss.item(),total_train_step)

    #test
    tudui.eval()
    total_test_loss=0
    total_accuracy=0
    with torch.no_grad():
        for data in test_dataloader:
            imgs , targets = data
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
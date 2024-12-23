import torchvision
from torch.utils.data import DataLoader

def load_data():
    train_data=torchvision.datasets.CIFAR10(root='./data', train=True, download=False,transform=torchvision.transforms.ToTensor())
    test_data=torchvision.datasets.CIFAR10(root='./data', train=False, download=False,transform=torchvision.transforms.ToTensor())
    train_data_size= len(train_data)
    test_data_size=len(test_data)
    print("train size == {} ".format(train_data_size) )
    print("test size == {} ".format(test_data_size) )
    train_dataloader = DataLoader(train_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)
    return train_data_size,test_data_size,train_dataloader,test_dataloader

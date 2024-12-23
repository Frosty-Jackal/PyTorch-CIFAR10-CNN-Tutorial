import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn

image_path = "data/dog.png"

image = Image.open(image_path)

image=image.convert("RGB")

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
print(image.shape)

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

model1 = torch.load("tudui_9_gpu.pth",map_location=torch.device('cpu'))
print(model1)
image=torch.reshape(image,(1,3,32,32))
model1.eval()
with torch.no_grad():
    output=model1(image)
    print(output.argmax(1))
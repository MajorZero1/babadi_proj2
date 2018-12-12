import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import csv
from run_network import run_net


train_batch_size = 50
test_batch_size = 100
num_of_epochs = 20
device = 'cuda'

train_log = './mnist_lenet_logs/train_log.csv'
test_log = './mnist_lenet_logs/test_log.csv'


#data loaders
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./MNIST_data', train=True, download=True,
                  transform=transforms.Compose([transforms.ToTensor()])),
            batch_size=train_batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./MNIST_data', train=False,
                  transform=transforms.Compose([transforms.ToTensor()])),
            batch_size=test_batch_size, shuffle=False)
            
#network
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.features = nn.Sequential( nn.Conv2d(1, 6, kernel_size=3),
                                  nn.ReLU(True),
                                  nn.MaxPool2d(2),
                                  nn.Conv2d(6, 16, kernel_size=3),
                                  nn.ReLU(True),
                                  nn.MaxPool2d(2) )
        self.fc = nn.Sequential( nn.Linear(400,120),
                                 nn.ReLU(True),
                                 nn.Linear(120, 84),
                                 nn.ReLU(True),
                                 nn.Linear(84, 10) )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 400)
        x = self.fc(x)
        return x
        
        

#setup csv files for logging
fieldnames = ['epoch','batch','loss','accuracy','batch_size']

train_csv = open(train_log,'w')
train_writer = csv.DictWriter(train_csv,fieldnames = fieldnames)
train_writer.writeheader()

test_csv = open(test_log,'w')
test_writer = csv.DictWriter(test_csv,fieldnames = fieldnames)
test_writer.writeheader()

#setup model and optimizer
model = LeNet().to(device)
optimizer = optim.Adam(model.parameters())        

#training and evaluation
for epoch in range(0,num_of_epochs):
     run_net(model,'train',epoch,train_loader,train_writer,device,optimizer=optimizer)
     run_net(model,'test',epoch,test_loader,test_writer,device)
         
         
train_csv.close()
test_csv.close()

#torch.save({'state_dict': model.state_dict()},'./lenet.pth')


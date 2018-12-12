import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import csv
from run_network import run_net
from monkey_dataset import MonkeyDataset


#any settings, where to output csvs with training/testing information
train_batch_size = 50
test_batch_size = 100
num_of_epochs = 20 
device = 'cuda'
train_log = './monkey_lenet_logs/train_log.csv'
test_log = './monkey_lenet_logs/test_log.csv'


training_data_path = '/scratch1/Daniel/babadiProj2/training'
testing_data_path = '/scratch1/Daniel/babadiProj2/validation'

#for resizing images and such
transform = transforms.Compose(
                 [transforms.ToPILImage(),
                 transforms.Resize((224,224), interpolation=2),
                 transforms.ToTensor()])

#datasets
train_dataset = MonkeyDataset(training_data_path, transform=transform)
test_dataset = MonkeyDataset(testing_data_path, transform=transform)

#data loaders
train_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size=train_batch_size, shuffle=True, num_workers=4)
        
test_loader = torch.utils.data.DataLoader(test_dataset,
            batch_size=test_batch_size, shuffle=False, num_workers=4)
            
#network
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3),
                                  nn.ReLU(True),
                                  nn.MaxPool2d(3),
                                  nn.Conv2d(16, 32, kernel_size=3),
                                  nn.ReLU(True),
                                  nn.MaxPool2d(3),
                                  nn.Conv2d(32,64, kernel_size=3),
                                  nn.ReLU(True),
                                  nn.MaxPool2d(3))
        self.fc = nn.Sequential( nn.Linear(3136,500),
                                 nn.ReLU(True),
                                 nn.Linear(500, 84),
                                 nn.ReLU(True),
                                 nn.Linear(84, 10) )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 3136)
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


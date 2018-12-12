import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import csv
from monkey_dataset import MonkeyDataset
from run_network import run_net

#any settings, where to output csvs with training/testing information
train_batch_size = 50
test_batch_size = 100
num_of_epochs = 10
num_of_ft_epochs = 10
device = 'cuda'

train_log = './monkey_transfer_logs/train_log.csv'
test_log = './monkey_transfer_logs/test_log.csv'
train_log_finetune='./monkey_transfer_logs/train_log_finetune.csv'
test_log_finetune='./monkey_transfer_logs/test_log_finetune.csv'

training_data_path = '/scratch1/Daniel/babadiProj2/training'
testing_data_path = '/scratch1/Daniel/babadiProj2/validation'


#for resizing images and such
transform = transforms.Compose(
                 [transforms.ToPILImage(),
                 transforms.Resize((224,224), interpolation=2),
                 transforms.ToTensor()])

#data loaders
train_loader = torch.utils.data.DataLoader(
    MonkeyDataset(training_data_path,
            transform=transform),
            batch_size=train_batch_size, shuffle=True, num_workers=4)
        
test_loader = torch.utils.data.DataLoader(
    MonkeyDataset(testing_data_path, 
            transform=transform),
            batch_size=test_batch_size, shuffle=False, num_workers=4)

#get a pretrained resnet18   
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
    
# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

model = model.to(device)

#setup csv files for logging
fieldnames = ['epoch','batch','loss','accuracy','batch_size']
train_csv = open(train_log,'w')
train_writer = csv.DictWriter(train_csv,fieldnames = fieldnames)
train_writer.writeheader()

test_csv = open(test_log,'w')
test_writer = csv.DictWriter(test_csv,fieldnames = fieldnames)
test_writer.writeheader()

optimizer = optim.Adam(model.fc.parameters())


for epoch in range(0, num_of_epochs):  	
    run_net(model,'train',epoch,train_loader,train_writer,device,optimizer=optimizer)
    run_net(model,'test',epoch,test_loader,test_writer,device)

train_csv.close()
test_csv.close()
###finetune the network###
    
#unfreaze the parameters
for param in model.parameters():
    param.requires_grad = True

#setup csv files for logging
train_csv = open(train_log_finetune,'w')
train_writer = csv.DictWriter(train_csv,fieldnames = fieldnames)
train_writer.writeheader()

test_csv = open(test_log_finetune,'w')
test_writer = csv.DictWriter(test_csv,fieldnames = fieldnames)
test_writer.writeheader()

optimizer = optim.Adam(model.parameters(),lr=0.0001)
for epoch in range(0, num_of_ft_epochs):
    run_net(model,'train',epoch,train_loader,train_writer,device,optimizer=optimizer)
    run_net(model,'test',epoch,test_loader,test_writer,device)

train_csv.close()
test_csv.close()


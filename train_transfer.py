import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import csv
from monkey_dataset import MonkeyDataset

#any settings, where to output csvs with training/testing information
train_batch_size = 50
test_batch_size = 100
num_of_epochs = 10
num_of_ft_epochs = 5
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
train_csv = open(train_log,'w')
fieldnames = ['epoch','batch','loss','accuracy']
train_writer = csv.DictWriter(train_csv,fieldnames = fieldnames)
train_writer.writeheader()

test_csv = open(test_log,'w')
test_writer = csv.DictWriter(test_csv,fieldnames = fieldnames)
test_writer.writeheader()

optimizer = optim.Adam(model.fc.parameters())
for epoch in range(0, num_of_epochs):  	
    #train an epoch
    model.train()
    correct = 0
    for batch_idx, (image, label) in enumerate(train_loader):
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(image)
        loss = F.cross_entropy(output,label)
        unused, predicted = output.max(1)
        correct_batch = predicted.eq(label.view_as(predicted)).sum().item()
        correct += correct_batch
        loss.backward()
        optimizer.step()
        
        if batch_idx % 2 == 0:
            print('Train epoch: %d \t Iter: %d \t Loss: \t %f' %
             (epoch, batch_idx, loss.item()))
             
        train_writer.writerow({'epoch': epoch, 'batch': batch_idx,
               'loss': loss.item(),'accuracy': correct_batch/train_batch_size})

    #test an epoch
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(test_loader):
            image, label = image.to(device), label.to(device)
            output = model(image)
            loss = F.cross_entropy(output,label)
            unused, predicted = output.max(1)
            correct_batch = predicted.eq(label.view_as(predicted)).sum().item()
            correct += correct_batch
            
            print('Test epoch: %d \t Iter: %d \t Loss: %f \t Correct: %d/%d' %
               (epoch, batch_idx, loss, correct_batch, test_batch_size))
               
            test_writer.writerow({'epoch': epoch, 'batch': batch_idx,
               'loss': loss.item(), 'accuracy': correct_batch/test_batch_size})
               
        print('Test epoch: %d \t Num correct: %d / %d' % 
         (epoch, correct, test_batch_size*len(test_loader)))

###finetune the network###

#unfreaze the parameters
for param in model.parameters():
    param.requires_grad = True


#setup csv files for logging
train_csv = open(train_log_finetune,'w')
fieldnames = ['epoch','batch','loss','accuracy']
train_writer = csv.DictWriter(train_csv,fieldnames = fieldnames)
train_writer.writeheader()

test_csv = open(test_log_finetune,'w')
test_writer = csv.DictWriter(test_csv,fieldnames = fieldnames)
test_writer.writeheader()

optimizer = optim.Adam(model.parameters())
for epoch in range(0, num_of_ft_epochs):
    #train an epoch
    model.train()
    correct = 0
    for batch_idx, (image, label) in enumerate(train_loader):
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(image)
        loss = F.cross_entropy(output,label)
        unused, predicted = output.max(1)
        correct_batch = predicted.eq(label.view_as(predicted)).sum().item()
        correct += correct_batch
        loss.backward()
        optimizer.step()

        if batch_idx % 2 == 0:
            print('Train finetune epoch: %d \t Iter: %d \t Loss: \t %f' %
             (epoch, batch_idx, loss.item()))

        train_writer.writerow({'epoch': epoch, 'batch': batch_idx,
               'loss': loss.item(),'accuracy': correct_batch/train_batch_size})

    #test an epoch
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(test_loader):
            image, label = image.to(device), label.to(device)
            output = model(image)
            loss = F.cross_entropy(output,label)
            unused, predicted = output.max(1)
            correct_batch = predicted.eq(label.view_as(predicted)).sum().item()
            correct += correct_batch

            print('Test finetune epoch: %d \t Iter: %d \t Loss: %f \t Correct: %d/%d' %
               (epoch, batch_idx, loss, correct_batch, test_batch_size))

            test_writer.writerow({'epoch': epoch, 'batch': batch_idx,
               'loss': loss.item(), 'accuracy': correct_batch/test_batch_size})

        print('Test epoch: %d \t Num correct: %d / %d' %
         (epoch, correct, test_batch_size*len(test_loader)))






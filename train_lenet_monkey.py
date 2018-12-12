import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import csv
from monkey_dataset import MonkeyDataset

#any settings, where to output csvs with training/testing information
train_batch_size = 50
test_batch_size = 100
num_of_epochs = 10 
device = 'cuda'
train_log = './monkey_lenet_logs/train_log.csv'
test_log = './monkey_lenet_logs/test_log.csv'
training_data_path = '/scratch1/Daniel/babadiProj2/training'
testing_data_path = '/scratch1/Daniel/babadiProj2/validation'


#for resizing images and such
transform = transforms.Compose(
                 [transforms.ToPILImage(),
                 transforms.Resize((28,28), interpolation=2),
                 transforms.ToTensor()])

#data loaders
train_loader = torch.utils.data.DataLoader(
    MonkeyDataset(training_data_path,
            transform=transform),
            batch_size=train_batch_size, shuffle=True)
        
test_loader = torch.utils.data.DataLoader(
    MonkeyDataset(testing_data_path, 
            transform=transform),
            batch_size=test_batch_size, shuffle=False)
            
#network
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.features = nn.Sequential( nn.Conv2d(3, 6, kernel_size=3),
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
train_csv = open(train_log,'w')
fieldnames = ['epoch','batch','loss','accuracy']
train_writer = csv.DictWriter(train_csv,fieldnames = fieldnames)
train_writer.writeheader()

test_csv = open(test_log,'w')
test_writer = csv.DictWriter(test_csv,fieldnames = fieldnames)
test_writer.writeheader()

#setup model and optimizer
model = LeNet().to(device)
optimizer = optim.Adam(model.parameters())

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
         
#torch.save({'state_dict': model.state_dict()},'./lenet.pth')
        
        

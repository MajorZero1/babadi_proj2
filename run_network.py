#train or evaluation the network for an epoch
def run_net(model, mode, epoch, data_loader, csv_writer, optimizer=None)	
    if mode == 'train':
    	model.train()
    	if optimizer == None
    	    raise Exception('must provide an optimizer in train mode')
    elif mode == 'test':
        print('in test mode')
        model.eval()
    else:
        raise Exception('mode must be test or train')
        
    total_correct = 0
    for batch_idx, (image, label) in enumerate(data_loader):
        image, label = image.to(device), label.to(device)
        if mode == 'train':
            optimizer.zero_grad()
            
        output = model(image)
        loss = F.cross_entropy(output,label)
        unused, predicted = output.max(1)
        correct_batch = predicted.eq(label.view_as(predicted)).sum().item()
        total_correct += correct_batch
        accuracy = correct_batch/image.size(0)
        if mode == 'train':
            loss.backward()
            optimizer.step()
        
        if batch_idx % 2 == 0:
            print('Mode %s \t Epoch: %d \t Iter: %d \t Loss: \t %f \t Accuracy %f' %
             (mode, epoch, batch_idx, loss.item(), accuracy))
             
        csv_writer.writerow({'epoch': epoch, 'batch': batch_idx,
               'loss': loss.item(),'accuracy': accuracy})
     print('end of epoch %d/%d correct' % (total_correct,len(data_loader.dataset)))
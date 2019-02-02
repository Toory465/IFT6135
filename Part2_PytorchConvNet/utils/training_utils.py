import torch
import torch.nn as nn

def train_model(model, optimizer, loader_train, loader_val, num_epochs, print_every, device):
    """
    Trains model on MNIST using optimizer object.
    
    Inputs:
    - model: Module object which is our model
    - optimizer: Optimizer object for training the model
    - num_epochs: Number of epochs to train
    
    Returns:
    - loss_hist: History of loss
    """
    model = model.to(device=device)  # Put model parameters on GPU or CPU
    criterion = nn.CrossEntropyLoss() # Define the loss function
    train_history = {'train_loss_hist':[], 'val_loss_hist':[],
                     'train_acc_hist':[], 'val_acc_hist':[]}
    
    iteration_count = 0
    it = 0
    for epoch in range(num_epochs):
        iteration_count += it
        for it, (X, y) in enumerate(loader_train):
            # Move inputs to specified device
            X = X.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)
            
            # Forward pass
            model.train()
            scores = model(X)
            train_loss = criterion(scores, y)
            
            # Backward pass
            optimizer.zero_grad() # Zero out the gradients
            train_loss.backward() # Compute gradient of loss w.r.t. model parameters
            optimizer.step() # Make a gradient update step

            if (it+iteration_count) % print_every == 0:
                ### Update train_history
                # train_loss_hist & train_acc_hist
                _, preds = scores.max(dim=1)
                train_acc = 100*(float((scores.max(dim=1)[1] == y).sum()) / y.size(0))
                train_history['train_loss_hist'].append(train_loss.item())
                train_history['train_acc_hist'].append(train_acc)
                # val_loss_hist & val_acc_hist
                val_acc, val_loss = evaluation(model, loader_val, criterion, device=device)
                train_history['val_loss_hist'].append(val_loss.item())
                train_history['val_acc_hist'].append(val_acc)

                # Print training process
                print('Epoch %d, Iteration %d:' % (epoch+1, it+iteration_count))
                print('Training data: loss = %.4f, accuracy = %.2f' % (train_loss.item(), train_acc))
                print('Validation data: loss = %.4f, accuracy = %.2f\n' % (val_loss.item(), val_acc))
        print('-----------')
    return train_history


def evaluation(model, loader, criterion, device):
    """
    Returns the accuracy of the model on loader data.
    
    Inputs:
    model: Module object which is our model
    loader: DataLoader object
    
    Returns:
    Accuracy percentage
    """
    num_correct = 0
    num_total = 0
    loss = 0
    model.eval()  # change model mode to eval
    with torch.no_grad():  # temporarily set all requires_grad flags to False
        for X, y in loader:
            # Move inputs to specified device
            X = X.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)
            
            # Compute scores (Forward pass)
            scores = model(X)
            _, preds = scores.max(dim=1)

            # Compute accuracy
            num_correct += (preds == y).sum()
            num_total += preds.size(0)
            if criterion is not None:
                # Compute loss
                loss += preds.size(0) * criterion(scores, y)
    acc = 100*(float(num_correct) / num_total)
    loss /= num_total
    return acc, loss
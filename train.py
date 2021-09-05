import numpy as np
# import matplotlib.pyplot as plt
import torch
# import torch.nn.functional as F
import argparse

# from PIL import Image
from torch import nn, optim
# from torchvision import datasets, transforms, models
# from collections import OrderedDict
from utils import *


# Create parse to get inputs in the command line
parser = argparse.ArgumentParser()


parser.add_argument('data_dir', type = str, default = 'flowers/', help = 'path to training data')
parser.add_argument('--save_dir', type = str, default = './', help = 'path to save trained model')
parser.add_argument('--arch', type = str, default = 'vgg', help = 'model architecture: choose vgg or densenet')
parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'learning rate of the optimizer')
parser.add_argument('--epochs', type = int, default = 7, help = 'number of epochs')
parser.add_argument('--hidden_units', type = int, default = 1024, help = 'number of hidden units')
parser.add_argument('--gpu', action='store_true', default=False)


# Save the input arguments in args
in_args = parser.parse_args()


# Load data for model training                    
data_sets, dataloaders = load_data(in_args.data_dir)

                    
if in_args.gpu:                   
    # Use GPU if it is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    # Use CPU
    device = torch.device("cpu")

model = get_model(in_args.arch, in_args.hidden_units)                   


# Define loss and optimizer.
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=in_args.learning_rate)

model.to(device)                    

# Train the network
epochs = in_args.epochs
steps = 0
running_loss = 0
                    
# Steps before printing validatn loss
print_every = 10

print('Training has commenced')
# Iterations
for epoch in range(epochs):
    for images, labels in dataloaders['train']:
        steps += 1

        # Move images and labels to the device(GPU), train train on train_datasets
        images, labels = images.to(device), labels.to(device)

        # Clear the gradients
        optimizer.zero_grad()

        # Obtain log probabilities, loss from the model, do a backward pass, step the optimizer
        logps = model(images)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        # Keep track of training loss
        running_loss += loss.item()

        # Drop out of the training loop to test network accuracy and loss on valid_datasets
        if steps % print_every == 0:
            accuracy = 0
            valid_loss = 0

            # Turn on evaluation inference mode: turn off dropouts, improve prediction accuracy
            model.eval()

            for images, labels in dataloaders['valid']:
                # Move images and labels to the device(GPU), train train on train_datasets
                images, labels = images.to(device), labels.to(device)

                logps = model(images)
                loss = criterion(logps, labels)
                valid_loss += loss.item()

                # Calculate class probabilities(ps), accuracy
                ps = torch.exp(logps)
                top_ps, top_class = ps.topk(1, dim=1)
                equality = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

            print(f"Epoch: {epoch + 1}/{epochs}  "
                  f"Train loss: {running_loss/print_every:.3f}  "
                  f"Validation loss: {valid_loss/len(dataloaders['valid']):.3f}  "
                  f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}  ")

            running_loss = 0
            model.train()

print('Training has ended')
# Save the checkpoint 
model.class_to_idx = data_sets['train'].class_to_idx
                    
checkpoint = {
              'hidden_units': in_args.hidden_units,
              'epochs': epochs,
              'model_arch': in_args.arch,
              'classifier': model.classifier,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx
             }
                    
print('Saving Model...')
torch.save(checkpoint, 'checkpoint.pth')
import numpy as np
import torch

from PIL import Image
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict



def load_data(data_dir):
    """
    Loads datasets from a directory
    
    Args:
        data_dir - directory where the datasets are
     
    Returns:
        datasets - datasets with transforms applied 
        dataloaders - iterable to loaad or sample the datasets 
    """
    # Data 
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Transforms for the training, validation, and testing sets
    train_transform = transforms.Compose([ transforms.RandomRotation(45),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225]) 
                                        ])

    valid_test_transform = transforms.Compose([ transforms.Resize(255), 
                                        transforms.CenterCrop(224), 
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])
                                        ])

    # Transform the datasets
    data_sets = {
        'train': datasets.ImageFolder(train_dir, transform=train_transform),
        'valid': datasets.ImageFolder(valid_dir, transform=valid_test_transform),
        'test': datasets.ImageFolder(test_dir, transform=valid_test_transform)
    } 

    # Define the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(data_sets['train'], batch_size=32, shuffle=True),
        'valid': torch.utils.data.DataLoader(data_sets['valid'], batch_size=32, shuffle=True),
        'test': torch.utils.data.DataLoader(data_sets['test'], batch_size=32, shuffle=True)
    } 

    return data_sets, dataloaders
    


def get_model(arch, hid_units):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
        
    Args:
        arch - architecture of the required model
        hid_units - number of feature units in the hidden layer
        
    returns:
        Model - a torchvision model with connected layers
    """
    
    # Models to choose from
    densenet161 = models.densenet161(pretrained = True)
    #     resnet152 = models.resnet152(pretrained = True)
    vgg19 = models.vgg19(pretrained=True)
    
    # Input features for the selected models
    densenet_in = densenet161.classifier.in_features
    #     resnet_in = resnet152.fc.in_features
    vgg_in = vgg19.classifier[0].in_features

    # Define a dict of available models and classifiers input features
    models_dict = {'densenet': densenet161, 'vgg': vgg19}
    in_units = {'densenet': densenet_in, 'vgg': vgg_in}

    
    # Model classifier
    classifier = nn.Sequential(
        OrderedDict([
            ('fc1', nn.Linear(in_units[arch], hid_units)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p=0.2)),
            ('fc2', nn.Linear(hid_units, 256)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(p=0.2)),
            ('fc3', nn.Linear(256, 102)),
            ('output', nn.LogSoftmax(dim=1)),
        ])
    )

    # Get the model
    model = models_dict[arch]
    
    # Turn off gradient     
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    # Define a new classifier for model (re)training
    # model.classifier = classifiers_dict[arch]
    model.classifier = classifier
    
    #     print(model.classifier)
    
    return model



def process_image(image_path):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
        
    Args:
        image_path - directory where the image is stored      
        
    returns:
        tensor_img - an Numpy array
    """
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image_path) 
    
    # Preprocess the image 
    preprocess = transforms.Compose([ transforms.Resize(255), 
                                       transforms.CenterCrop(224), 
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])
                                     ])
    
    # Convert normalized image to NP array and then to a tensor image
    
    np_img = np.array(preprocess(img))
    tensor_img = torch.from_numpy(np_img)
    
    return tensor_img



def load_checkpoint(file_path):
    """ 
    Loads a checkpoint and rebuilds the models,
    
    Args:
        filepath - directory where checkpoint file is stored    
    
    returns:
        model - a model rebuilt from a checkpoint
    """
    
    checkpoint = torch.load(file_path)

    model = get_model(checkpoint['model_arch'], checkpoint['hidden_units'])
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    print(model)
    
    return model


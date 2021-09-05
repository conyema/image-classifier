import numpy as np
# import matplotlib.pyplot as plt
import torch
# import torch.nn.functional as F
import argparse
import json

# from PIL import Image
# from torch import nn, optim
from utils import *


# Create parse to get inputs in the command line
parser = argparse.ArgumentParser()


parser.add_argument('image_path', type = str, help = 'path to an input image')
parser.add_argument('checkpoint', type = str, default = 'checkpoint.pth', help = 'path to saved checkpoint of trained model')
parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'mapping of categories to real names')
parser.add_argument('--top_k', type = int, default = 5, help = 'top k number of classes to return: where k is an integer')
parser.add_argument('--gpu', action='store_true', default=False)


# Save the input arguments in args
in_args = parser.parse_args()

# Get mapping of categories to real names
with open(in_args.category_names, 'r') as f:
    cat_to_name = json.load(f)



if in_args.gpu:                   
    # Use GPU if it is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    # Use CPU
    device = torch.device("cpu")

# Load the checkpoint
model = load_checkpoint(in_args.checkpoint)

# Process image and correct input data dimension
tensor_img = process_image(in_args.image_path).unsqueeze(0)

# Move image and model to the device(GPU)
image = tensor_img.to(device)
model = model.to(device)

# Turn on evaluation inference mode
model.eval()


logps = model(image)

# Calculate class probabilities(ps)
ps = torch.exp(logps)
top_ps, top_class = ps.topk(in_args.top_k, dim=1)

probs = top_ps.tolist()[0]
classes = top_class.tolist()[0]

# invert class_to_idx: use index 
idx_to_class = {val: key for key, val in model.class_to_idx.items()}
class_keys = [idx_to_class[val] for val in classes]

# Obtain the real names of the classes
flowers = [cat_to_name[key] for key in class_keys]

print (f"Probability: {probs}  "
       f"Class: {classes}  "
       f"Result: {flowers}  "
      )
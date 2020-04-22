import os 
import numpy as np 
import argparse 
import math 
import itertools 

import torch 
import torch.nn as nn
from torch.utils import data
import torch.nn.functional
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm 
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description = "Configuration parameters for training and testing the Schtochastic TravelGAN") 

parser.add_argument('--train', action = 'store_true') 
parser.add_argument('--image', type = str) 
parser.add_argument('--model', type = str, required = True) 

args = parser.parse_args()
print(args) 










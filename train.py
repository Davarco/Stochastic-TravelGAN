import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torch.nn.functional
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import cv2

from model import *
from data import *

def train(X, Y, gan):
    X_size = X.shape[0]
    Y_size = Y.shape[0]

    epochs = 5
    batch_size = 8
    cv2.waitKey(0)

def main():
    X, Y = get_fruit_data()
    X = X[:500]
    Y = Y[:500]
    print('X Shape:', X.shape)
    print('Y Shape:', Y.shape)
    
    gan = TravelGAN()
    train(X, Y, gan)

if __name__ == '__main__':
    main()

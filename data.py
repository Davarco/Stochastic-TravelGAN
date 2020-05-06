import torch 
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torchvision.transforms import ToTensor, Normalize, Resize
from torchvision.transforms import Compose
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

class CifarClass(Dataset): 
    #for path use '~/.torch/data' 
    def __init__(self, path, imageClass, classSize): 
        dataset = CIFAR10(path, download=True)
        mappings = np.where(np.array(dataset.targets) == dataset.class_to_idx[imageClass])  
        
        self.dataSubset = [dataset.data[x] for x in mappings][0]
        self.dataSubset = self.dataSubset[:classSize] 
        
        self.transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
    def __getitem__(self, index): 
        return self.transform(self.dataSubset[index]) 
    
    def __len__(self): 
        return len(self.dataSubset) 
    
    def visualize(tensor): 
        plt.imshow(np.transpose(tensor, (1, 2, 0)), interpolation='nearest') 

class MaleHatClass(Dataset):
    def __init__(self, size):
        df = pd.read_csv('data/celebA/list_attr_celeba.csv')
        hat = df.loc[(df['Male'] == 1) & (df['Wearing_Hat'] == 1)].values
        folder = 'data/celebA/img_align_celeba/img_align_celeba'
        self.files = []
        for i in range(size):
            img = Image.open('{}/{}'.format(folder, hat[i, 0]))
            img = transforms.functional.resize(img, (128, 128))
            img = transforms.ToTensor()(img).cuda()
            self.files.append(img)

    def __getitem__(self, index): 
        return self.files[index]
    
    def __len__(self): 
        return len(self.files) 

class MaleNoHatClass(Dataset):
    def __init__(self, size):
        df = pd.read_csv('data/celebA/list_attr_celeba.csv')
        nohat = df.loc[(df['Male'] == 1) & (df['Wearing_Hat'] == -1)].values
        folder = 'data/celebA/img_align_celeba/img_align_celeba'
        self.files = []
        for i in range(size):
            img = Image.open('{}/{}'.format(folder, nohat[i, 0]))
            img = transforms.functional.resize(img, (128, 128))
            img = transforms.ToTensor()(img).cuda()
            self.files.append(img)

    def __getitem__(self, index): 
        return self.files[index]
    
    def __len__(self): 
        return len(self.files) 

def get_fruit_data():
    apples = []
    oranges = []
    for i in range(984):
        img = cv2.imread('data/apples/{}.jpg'.format(i), 1) 
        if img is not None and abs(1-img.shape[0]/img.shape[1]) < 0.3: 
            apples.append(img)
    for i in range(684):
        img = cv2.imread('data/oranges/{}.jpg'.format(i), 1) 
        if img is not None and abs(1-img.shape[0]/img.shape[1]) < 0.35: 
            oranges.append(img)
    for i in range(len(apples)):
        apples[i] = cv2.resize(apples[i], (128, 128)).astype('float64')
        apples[i] /= 255
    for i in range(len(oranges)):
        oranges[i] = cv2.resize(oranges[i], (128, 128)).astype('float64')
        oranges[i] /= 255
    X = np.array(apples)
    Y = np.array(oranges)
    return X, Y

def get_celeb_data():
    df = pd.read_csv('data/celebA/list_attr_celeba.csv')
    male_hat = df.loc[(df['Male'] == 1) & (df['Wearing_Hat'] == 1)].values
    male_nohat = df.loc[(df['Male'] == 1) & (df['Wearing_Hat'] == -1)].values
    hat = []
    nohat = []
    folder = 'data/celebA/img_align_celeba/img_align_celeba'
    for i in range(1000):
        name = male_hat[i][0]
        img = cv2.imread('{}/{}'.format(folder, name), 1) 
        hat.append(img)
    for i in range(1000):
        name = male_nohat[i][0]
        img = cv2.imread('{}/{}'.format(folder, name), 1) 
        img = Image.open('{}/{}'.format(folder, name))
        nohat.append(img)
    for i in range(len(hat)):
        hat[i] = cv2.resize(hat[i], (128, 128)).astype('float64')
        hat[i] /= 255
    for i in range(len(nohat)):
        nohat[i] = cv2.resize(nohat[i], (128, 128)).astype('float64')
        nohat[i] /= 255
    X = np.array(hat)
    Y = np.array(nohat)
    return X, Y

def main():
    # X, Y = get_fruit_data()
    # print(X.shape, Y.shape)
    # cv2.imshow('X', X[0])
    # cv2.imshow('Y', Y[0])
    # cv2.waitKey(0)
    # X, Y = get_celeb_data()
    # print(X.shape, Y.shape)
    hat = MaleHatClass(1000)
    nohat = MaleNoHatClass(1000)

if __name__ == '__main__':
    main()

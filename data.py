from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torchvision.transforms import ToTensor, Normalize
from torchvision.transforms import Compose
import numpy as np
import matplotlib.pyplot as plt
import cv2

class CifarClass(Dataset): 
    #for path use '~/.torch/data' 
    def __init__(self, path, imageClass, classSize): 
        dataset = CIFAR10(path, download=True)
        mappings = np.where(np.array(dataset.targets) == data.class_to_idx[imageClass])  
        
        self.dataSubset = [dataset.data[x] for x in mappings][0]
        self.dataSubset = self.dataSubset[: classSize] 
        
        self.transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
    def __get__(self, index): 
        return self.transform(self.dataSubset[index]) 
    
    def __len__(self): 
        return len(self.dataSubset) 
    
    def visualize(tensor): 
        plt.imshow(np.transpose(tensor, (1, 2, 0)), interpolation = 'nearest') 

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
    X = np.zeros((len(apples), 128, 128, 3))
    Y = np.zeros((len(oranges), 128, 128, 3))
    for i in range(len(apples)):
        X[i] = apples[i].resize(128, 128, 3)
    for i in range(len(oranges)):
        X[i] = oranges[i].resize(128, 128, 3)
    return X, Y

def main():
    X, Y = get_fruit_data()
    print(X.shape, Y.shape)

if __name__ == '__main__':
    main()

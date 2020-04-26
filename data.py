from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torchvision.transforms import ToTensor, Normalize
from torchvision.transforms import Compose
import numpy as np
import matplotlib.pyplot as plt

class CifarClass(Dataset): 
    #for path use '~/.torch/data' 
    def __init__(self, path, imageClass, classSize): 
        dataset = CIFAR10(path, download = True)
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
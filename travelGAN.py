import torch
import torch.nn as nn
from torch.optim import Adam
from models import SiameseNetwork, Generator, Discriminator
import os

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()

print(opt)

class TravelGan(nn.module): 
    def __init__(self): 
        self.generatorAtoB = Generator() 
        self.generatorBtoA = Generator()
        
        self.discriminatorA = Discriminator() 
        self.discriminatorB = Discriminator()
        
        self.siamese = SiameseNetwork() 
        
        self.generatorOptimizer = Adam(list(self.generatorAtoB.parameters()) + list(self.generatorBtoA.parameters()) + list(self.siamese.parameters())) 
        self.discriminatorOptimizer = Adam(list(self.discriminatorA.parameters()) + list(self.discriminatorB.parameters()))
        
    def forward(self, a, b): 
        return (self.generatorAtoB(a), self.generatorBtoA(b))
    
    
travelGAN = TravelGAN() 

dataLoaderA = #fill in with dataloader for apples
dataLoaderB = #fill in with dataloader for oranges

for x in range(0, opt.n_epochs): 
    for (a_real, b_real) in enumerate(zip(dataLoaderA, dataLoaderB)):
        x_ab = self.generatorAtoB(a_real, b_real)
        x_ba = self.generatorBtoA(a_real, b_real) 

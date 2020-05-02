import torch 
import torch.nn as nn
from torch.utils import data
import torch.nn.functional
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.c1 = nn.Sequential(
                nn.Conv2d(3, 8, 3, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.MaxPool2d(2)
                )
        self.c2 = nn.Sequential(
                nn.Conv2d(8, 16, 3, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.MaxPool2d(2)
                )
        self.c3 = nn.Sequential(
                nn.Conv2d(16, 32, 3, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.MaxPool2d(2)
                )
        self.c4 = nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.MaxPool2d(2)
                )
        self.c5 = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.MaxPool2d(2)
                )
        self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(2048, 1),
                nn.Sigmoid()
                )

    def forward_pass(self, X):
        X = self.c1(X)
        X = self.c2(X)
        X = self.c3(X)
        X = self.c4(X)
        X = self.c5(X)
        X = self.fc(X)
        return X

    def forward(self, X):
        return self.forward_pass(X)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()


class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.c1 = nn.Sequential(
                nn.Conv2d(3, 8, 3, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.MaxPool2d(2)
                )
        self.c2 = nn.Sequential(
                nn.Conv2d(8, 16, 3, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.MaxPool2d(2)
                )
        self.c3 = nn.Sequential(
                nn.Conv2d(16, 32, 3, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.MaxPool2d(2)
                )
        self.c4 = nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.MaxPool2d(2)
                )
        self.c5 = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.MaxPool2d(2)
                )
        self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(2048, 1000),
                nn.Sigmoid()
                )

    def forward_pass(self, X):
        X = self.c1(X)
        X = self.c2(X)
        X = self.c3(X)
        X = self.c4(X)
        X = self.c5(X)
        X = self.fc(X)
        return X

    def forward(self, X1, X2):
        return self.forward_pass(X1), self.forward_pass(X2)

def main():
    X = torch.randn(1, 3, 128, 128)
    discriminator = Discriminator()
    X = discriminator.forward(X)
    print('Discriminator Output Shape:', X.shape)

    X = torch.randn(1, 3, 128, 128)
    siamese = Siamese()
    X, X = siamese.forward(X, X)
    print('Siamese Output Shape', X.shape)

if __name__ == '__main__':
    main()

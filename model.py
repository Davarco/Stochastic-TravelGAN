import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torch.nn.functional
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt

class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.c1 = conv_block(3, 8, 3, 2, 1, True)
        self.c2 = conv_block(8, 16, 3, 2, 1, True)
        self.c3 = conv_block(16, 32, 3, 2, 1, True)
        self.c4 = conv_block(32, 64, 3, 2, 1, True)
        self.c5 = conv_block(64, 128, 3, 2, 1, True)
        self.fc = linear_block(2048, 1000, nn.Sigmoid)

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

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.c1 = conv_block(3, 8, 3, 2, 1, False)
        self.c2 = conv_block(8, 16, 3, 2, 1, True)
        self.c3 = conv_block(16, 32, 3, 2, 1, True)
        self.c4 = conv_block(32, 64, 3, 2, 1, True)
        self.c5 = conv_block(64, 128, 3, 2, 1, True)
        self.fc = linear_block(2048, 1, nn.Sigmoid)

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
        self.c1 = conv_block(3, 8, 3, 2, 1, True)
        self.c2 = conv_block(8, 16, 3, 2, 1, True)
        self.c3 = conv_block(16, 32, 3, 2, 1, True)
        self.c4 = conv_block(32, 64, 3, 2, 1, True)
        self.c5 = conv_block(64, 128, 3, 2, 1, True)
        self.t1 = uconv_block(128, 64, 3, 2, 1, 1)
        self.tc1 = conv_block(128, 64, 3, 1, 1, True)
        self.t2 = uconv_block(64, 32, 3, 2, 1, 1)
        self.tc2 = conv_block(64, 32, 3, 1, 1, True)
        self.t3 = uconv_block(32, 16, 3, 2, 1, 1)
        self.tc3 = conv_block(32, 16, 3, 1, 1, True)
        self.t4 = uconv_block(16, 8, 3, 2, 1, 1)
        self.tc4 = conv_block(16, 8, 3, 1, 1, True)
        self.t5 = uconv_block(8, 3, 3, 2, 1, 1)
        self.tc5 = uconv_block_final(6, 3, 3, 1, 1, True)

    def forward_pass(self, X):
        X_init = X
        X_c1 = self.c1(X)
        X_c2 = self.c2(X_c1)
        X_c3 = self.c3(X_c2)
        X_c4 = self.c4(X_c3)
        X_c5 = self.c5(X_c4)

        X = torch.cat((self.t1(X_c5), X_c4), 1)
        X = self.tc1(X)
        X = torch.cat((self.t2(X), X_c3), 1)
        X = self.tc2(X)
        X = torch.cat((self.t3(X), X_c2), 1)
        X = self.tc3(X)
        X = torch.cat((self.t4(X), X_c1), 1)
        X = self.tc4(X)
        X = torch.cat((self.t5(X), X_init), 1)
        X = self.tc5(X)

        return X

    def forward(self, X):
        return self.forward_pass(X)

def conv_block(in_ch, out_ch, kernel, stride, padding, batch_norm):
    if batch_norm:
        return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel, stride, padding),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, True)
                )
    else:
        return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel, stride, padding),
                nn.LeakyReLU(0.2, True)
                )

def uconv_block(in_ch, out_ch, kernel, stride, padding, output_padding):
    return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding, 
                output_padding),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, True)
            )

def uconv_block_final(in_ch, out_ch, kernel, stride, padding, batch_norm):
    return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.Tanh()
            )

def linear_block(in_ch, out_ch, activation):
    return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, out_ch),
            activation()
            )

def main():
    X = torch.randn(1, 3, 128, 128)
    generator = Generator()
    X = generator.forward(X)
    print('Generator Output Shape:', X.shape)

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

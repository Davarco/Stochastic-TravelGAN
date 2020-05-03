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
        self.c1 = conv_block(3, 8, 3, 2, 1, False)
        self.c2 = conv_block(8, 16, 3, 2, 1, True)
        self.c3 = conv_block(16, 32, 3, 2, 1, True)
        self.c4 = conv_block(32, 64, 3, 2, 1, True)
        self.c5 = conv_block(64, 128, 3, 2, 1, True)
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
        self.c1 = conv_block(3, 8, 3, 2, 1, True)
        self.c2 = conv_block(8, 16, 3, 2, 1, True)
        self.c3 = conv_block(16, 32, 3, 2, 1, True)
        self.c4 = conv_block(32, 64, 3, 2, 1, True)
        self.c5 = conv_block(64, 128, 3, 2, 1, True)
        self.t1 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1,
                    output_padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, True),
                )
        self.t2 = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1,
                    output_padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2, True),
                )
        self.t3 = nn.Sequential(
                nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1,
                    output_padding=1),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(0.2, True),
                )
        self.t4 = nn.Sequential(
                nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1,
                    output_padding=1),
                nn.BatchNorm2d(8),
                nn.LeakyReLU(0.2, True),
                )
        self.t5 = nn.Sequential(
                nn.ConvTranspose2d(8, 3, 3, stride=2, padding=1,
                    output_padding=1),
                nn.BatchNorm2d(3),
                nn.LeakyReLU(0.2, True),
                )

    def forward_pass(self, X):
        X_c1 = self.c1(X)
        print('c1', X_c1.shape)

        X_c2 = self.c2(X_c1)
        print('c2', X_c2.shape)

        X_c3 = self.c3(X_c2)
        print('c3', X_c3.shape)

        X_c4 = self.c4(X_c3)
        print('c4', X_c4.shape)

        X_c5 = self.c5(X_c4)
        print('c5', X_c5.shape)

        X_t1 = torch.cat((self.t1(X_c5), X_c4), 1)
        print('t1', X_t1.shape)

        # X_t2 = self.t2(X_t1)
        # print('t2', X_t2.shape)
        # X_t3 = self.t3(X_t2)
        # print('t3', X_t3.shape)
        # X_t4 = self.t4(X_t3)
        # print('t4', X_t4.shape)
        # X_t5 = self.t5(X_t4)
        # print('t5', X_t5.shape)

        return X_t5

    def forward(self, X):
        return self.forward_pass(X)

class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.c1 = conv_block(3, 8, 3, 2, 1, True)
        self.c2 = conv_block(8, 16, 3, 2, 1, True)
        self.c3 = conv_block(16, 32, 3, 2, 1, True)
        self.c4 = conv_block(32, 64, 3, 2, 1, True)
        self.c5 = conv_block(64, 128, 3, 2, 1, True)
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
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, True)
                )

def conv_block_transpose(in_ch, out_ch, kernel, stride, padding, batch_norm):
    if batch_norm:
        return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, True)
                )
    else:
        return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, True)
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
    print(X)

    X = torch.randn(1, 3, 128, 128)
    siamese = Siamese()
    X, X = siamese.forward(X, X)
    print('Siamese Output Shape', X.shape)

if __name__ == '__main__':
    main()

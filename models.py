import torch 
import torch.nn as nn
from torch.utils import data
import torch.nn.functional
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.cnn = nn.Sequential(
                nn.Conv2d(3, 64, 10),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 7),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 128, 4),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 4),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Flatten()
                )

        self.fc = nn.Sequential(
                nn.Linear(2304, 1000),
                nn.Sigmoid()
                )

    def forward(self, input1, input2):
        output1 = self.fc(self.cnn(input1))
        output2 = self.fc(self.cnn(input2))
        return output1, output2

class contrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label)*torch.pow(euclidean_distance, 2)+(label)*torch.pow(torch.clamp(self.margin-euclidean_distance, min=0.0), 2))
        return loss_contrastive

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


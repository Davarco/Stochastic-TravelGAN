from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
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

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label)*torch.pow(euclidean_distance, 2)+(label)*torch.pow(torch.clamp(self.margin-euclidean_distance, min=0.0), 2))
        return loss_contrastive

def get_data():
    images = np.zeros((964, 20, 3, 105, 105))
    for c in range(1, 965):
        for i in range(1, 21):
            img = cv2.imread('data/images/{}/{}.png'.format(c, i), 1)
            img = img.reshape(3, 105, 105)
            images[c-1][i-1] = img
    return images

def main():
    siamese = SiameseNetwork()
    criterion = ContrastiveLoss()
    optimizer = optim.SGD(siamese.parameters(), lr=0.00005)
    X_raw = get_data()
    for i in range(964):
        a = random.randint(0, 19)
        b = (a + random.randint(1, 19)) % 20
        c = (i + random.randint(1, 963)) % 964
        d = random.randint(0, 19)

        anchor = torch.Tensor(X_raw[i][a]).unsqueeze(0)
        genuine = torch.Tensor(X_raw[i][b]).unsqueeze(0)
        imposter = torch.Tensor(X_raw[c][d]).unsqueeze(0)
        # cv2.imshow('a', X_raw[i][a].reshape(105, 105, 3))
        # cv2.imshow('b', X_raw[i][b].reshape(105, 105, 3))
        # cv2.imshow('c', X_raw[c][d].reshape(105, 105, 3))
        # cv2.waitKey(0)

        g1, g2 = siamese(anchor, genuine)
        optimizer.zero_grad()
        loss_contrastive = criterion(g1, g2, 0)
        loss_contrastive.backward()
        optimizer.step()
        print(loss_contrastive.item())

        i1, i2 = siamese(anchor, imposter)
        optimizer.zero_grad()
        loss_contrastive = criterion(i1, i2, 1)
        loss_contrastive.backward()
        optimizer.step()
        print(loss_contrastive.item())

    print('Raw Data:', X_raw.shape)

if __name__ == "__main__":
    main()

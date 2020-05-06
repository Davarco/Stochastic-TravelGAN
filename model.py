import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torch.nn.functional
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

class TravelGAN(nn.Module):
    def __init__(self):
        super(TravelGAN, self).__init__()

        self.S = Siamese().cuda()
        self.G_XY = Generator().cuda()
        self.G_YX = Generator().cuda()
        self.DX = Discriminator().cuda()
        self.DY = Discriminator().cuda()
        
        SG_params = list(self.S.parameters()) + list(self.G_XY.parameters()) + \
                list(self.G_YX.parameters())
        D_params = list(self.DX.parameters()) + list(self.DY.parameters())
        self.SG_opt = optim.Adam(SG_params, lr=0.0002, betas=(0.5, 0.9))
        self.D_opt = optim.Adam(D_params, lr=0.0002, betas=(0.5, 0.9))

class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.c1 = conv_block(3, 64, 4, 2, 1, False)
        self.c2 = conv_block(64, 128, 4, 2, 1, True)
        self.c3 = conv_block(128, 256, 4, 2, 1, True)
        self.c4 = conv_block(256, 512, 4, 2, 1, True)
        self.c5 = conv_block(512, 512, 4, 2, 1, True)
        self.fc = linear_block(8192, 1000)

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

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.c1 = conv_block(3, 64, 4, 2, 1, False)
        self.c2 = conv_block(64, 128, 4, 2, 1, True)
        self.c3 = conv_block(128, 256, 4, 2, 1, True)
        self.c4 = conv_block(256, 256, 4, 2, 1, True)
        self.t1 = uconv_block(256, 256, 4, 2, 1, 0)
        self.t2 = uconv_block(512, 128, 4, 2, 1, 0)
        self.t3 = uconv_block(256, 64, 4, 2, 1, 0)
        self.t4 = uconv_block_final(128, 3, 4, 2, 1, 0)

    def forward_pass(self, X):
        print(X.shape)
        X_init = X
        X_c1 = self.c1(X)
        X_c2 = self.c2(X_c1)
        X_c3 = self.c3(X_c2)
        X_c4 = self.c4(X_c3)

        X = self.t1(X_c4)
        X = torch.cat((X_c3, X), 1)
        X = self.t2(X)
        X = torch.cat((X_c2, X), 1)
        X = self.t3(X)
        X = torch.cat((X_c1, X), 1)
        X = self.t4(X)

        return X

    def forward(self, X):
        return self.forward_pass(X)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.c1 = conv_block(3, 128, 4, 2, 1, False)
        self.c2 = conv_block(128, 256, 4, 2, 1, True)
        self.c3 = conv_block(256, 512, 4, 2, 1, True)
        self.c4 = conv_block(512, 1024, 4, 2, 1, True)
        self.c5 = conv_block(1024, 1024, 4, 2, 1, True)
        self.fc = linear_block(16384, 1)

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

def conv_block_final(in_ch, out_ch, kernel, stride, padding, batch_norm):
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
            # nn.LeakyReLU(0.2, True)
            nn.ReLU(True)
            )

def uconv_block_final(in_ch, out_ch, kernel, stride, padding, output_padding):
    return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding,
                output_padding),
            # nn.BatchNorm2d(out_ch),
            nn.Tanh()
            )

def linear_block(in_ch, out_ch):
        return nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_ch, out_ch)
                )

def adversarial_loss(logits, genuine):
    batch_size = logits.size(0)
    if genuine:
        labels = torch.ones(batch_size).cuda()
    else:
        labels = torch.zeros(batch_size).cuda()
    return nn.BCEWithLogitsLoss()(logits.squeeze(), labels)

def transformation_vector_loss(SX, SX_gen):
    pairs = np.asarray(list(combinations(range(SX.shape[0]), 2)))
    # V0 = F.normalize(SX[pairs[:, 0]] - SX[pairs[:, 1]], dim=1)
    # V1 = F.normalize(SX_gen[pairs[:, 0]] - SX_gen[pairs[:, 1]], dim=1)
    # return torch.mean(torch.sum(-V0*V1, dim=1))
    V0 = SX[pairs[:, 0]] - SX[pairs[:, 1]]
    V1 = SX_gen[pairs[:, 0]] - SX_gen[pairs[:, 1]]
    angle_dist = nn.CosineSimilarity()(V0, V1).mean()
    mag_dist = nn.MSELoss(reduction='mean')(V0, V1)
    return mag_dist - angle_dist

def contrastive_loss_same(SX, SX_gen):
    V = SX - SX_gen
    # V = V**2
    return torch.mean(V)

def contrastive_loss_different(SX):
    pairs = np.asarray(list(combinations(range(SX.shape[0]), 2)))
    V = SX[pairs[:, 0]] - SX[pairs[:, 1]]
    # V = V**2
    # same = 0
    # loss = (1-same)*torch.pow(V, 2) + same*torch.pow(torch.clamp(10-V, min=0), 2)
    loss = torch.clamp(10 - torch.norm(V, 1), min=0)
    return loss
    # return torch.mean(loss)

def main():
    # gan = TravelGAN()
    # X = torch.randn(5, 3, 128, 128).cuda()
    # Y = torch.randn(5, 3, 128, 128).cuda()
    # X_gen = gan.G_YX(Y)
    # Y_gen = gan.G_XY(X)

    # X_logits = gan.DX(X)
    # Y_logits = gan.DY(Y)
    # X_gen_logits = gan.DX(X_gen)
    # Y_gen_logits = gan.DY(Y_gen)

    # SX, SY = gan.S(X, Y)
    # SX_gen, SY_gen = gan.S(X_gen, Y_gen)

    # X_adv_loss = adversarial_loss(X_logits, True)
    # Y_adv_loss = adversarial_loss(Y_logits, True)
    # X_gen_adv_loss = adversarial_loss(X_gen_logits, False)
    # Y_gen_adv_loss = adversarial_loss(Y_gen_logits, False)

    # X_vec_loss = transformation_vector_loss(SX, SY, SX_gen, SY_gen)

    # X_con_loss = contrastive_loss(SX, SY, SX_gen, SY_gen, 1)

    # print(X_logits.shape)
    # print(Y_logits.shape)
    # print(X_gen_logits.shape)
    # print(Y_gen_logits.shape)
    # print(SX.shape)
    # print(SY.shape)
    # print(SX_gen.shape)
    # print(SY_gen.shape)
    # print(X_adv_loss)
    # print(Y_adv_loss)
    # print(X_gen_adv_loss)
    # print(Y_gen_adv_loss)
    # print(X_vec_loss)
    # print(X_con_loss)
       
    X = torch.randn(5, 3, 128, 128)
    siamese = Siamese()
    print(siamese)
    X, X = siamese.forward(X, X)
    X = torch.randn(5, 3, 128, 128)
    generator = Generator()
    print(generator)
    X = generator.forward(X)
    X = torch.randn(5, 3, 128, 128)
    discriminator = Discriminator()
    print(discriminator)
    X = discriminator.forward(X)

if __name__ == '__main__':
    main()

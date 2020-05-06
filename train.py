import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torch.nn.functional
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

from model import *
from data import *

def train(X_dataloader, Y_dataloader, gan):
    torch.autograd.set_detect_anomaly(True)

    epochs = 500
    steps = 0
    for e in range(epochs):
        for X, Y in zip(X_dataloader, Y_dataloader):
            # Train the discriminator.
            gan.D_opt.zero_grad()
            X_gen = gan.G_YX(Y)
            Y_gen = gan.G_XY(X)

            X_logits = gan.DX(X)
            Y_logits = gan.DY(Y)
            X_gen_logits = gan.DX(X_gen)
            Y_gen_logits = gan.DY(Y_gen)

            X_dis_loss = 0.5 * adversarial_loss(X_logits, True)
            Y_dis_loss = 0.5 * adversarial_loss(Y_logits, True)
            X_gen_dis_loss = 0.5 * adversarial_loss(X_gen_logits, False)
            Y_gen_dis_loss = 0.5 * adversarial_loss(Y_gen_logits, False)
            dis_loss = X_dis_loss + X_gen_dis_loss + Y_dis_loss + Y_gen_dis_loss
            d1 = X_dis_loss.item() + Y_dis_loss.item()
            d2 = X_gen_dis_loss.item() + Y_gen_dis_loss.item()

            dis_loss.backward()
            gan.D_opt.step()

            # Train the generator and siamese network.
            gan.SG_opt.zero_grad()

            X_gen = gan.G_YX(Y)
            Y_gen = gan.G_XY(X)
            X_gen_logits = gan.DX(X_gen)
            Y_gen_logits = gan.DY(Y_gen)

            SX, SY = gan.S(X, Y)
            SX_gen, SY_gen = gan.S(X_gen, Y_gen)

            X_gen_dis_loss = adversarial_loss(X_gen_logits, True)
            Y_gen_dis_loss = adversarial_loss(Y_gen_logits, True)
            vec_loss = 0.5 * transformation_vector_loss(SX, SX_gen)
            vec_loss += 0.5 * transformation_vector_loss(SY, SY_gen)
            siamese_diff = 0.5 * contrastive_loss_different(SX)
            siamese_diff += 0.5 * contrastive_loss_different(SY)
            siamese_same = 0.5 * contrastive_loss_same(SX, SX_gen)
            siamese_same += 0.5 * contrastive_loss_same(SY, SY_gen)

            gen_loss = X_gen_dis_loss + Y_gen_dis_loss
            gen_loss += 10 * (vec_loss + siamese_diff + siamese_same)
            gen_loss.backward()

            gan.SG_opt.step()

            d = dis_loss.item()
            g = gen_loss.item()
            t = d + g
            # print('Loss: (generator) {:<8.4f} (discriminator) '
            #         '{:<8.4f}'.format(combined_loss, dis_loss))

            g1 = X_gen_dis_loss.item()
            g2 = Y_gen_dis_loss.item()
            v = 10 * vec_loss.item()
            s1 = 10 * siamese_diff.item()
            s2 = 10 * siamese_same.item()

            steps += 1

            print('Loss: (t) {:<8.4f} (d) {:<8.4f} (g) {:<8.4f}'.format(t, d, g))
            print('\tGen: (X_gen) {:<8.4f} (Y_gen) {:<8.4f} (TraVeL) {:<8.4f} '
                    '(Siamese Diff) {:<8.4f} (Siamese Same) {:<8.4f}'
                    .format(g1, g2, v, s1, s2))

            # if i in check:
            if steps:
                out = open('losses.txt', 'a')
                out.write('{} {} {} {} {} {} {}\n'.format(d1, d2, g1, g2, s1, s2, v))
                out.close()

            if steps == 1 or steps % 100 == 0:
                disp_tensor_as_image(X[0], steps, 'X.jpg')
                disp_tensor_as_image(Y[0], steps, 'Y.jpg')
                disp_tensor_as_image(X_gen[0], steps, 'X_gen.jpg')
                disp_tensor_as_image(Y_gen[0], steps, 'Y_gen.jpg')
        # i += 1

def disp_tensor_as_image(X, step, name):
    img = transforms.Normalize((-1, -1, -1), (2, 2, 2))(X.cpu())
    img = transforms.ToPILImage()(img)
    img.save('output/{}_{}'.format(step, name))
    img.show()

def main():
    # X, Y = get_fruit_data()
    # birds = DataLoader(CifarClass('~/.torch/data', 'bird', 1000), batch_size=16,
    #         shuffle=True)
    # ships = DataLoader(CifarClass('~/.torch/data', 'ship', 1000), batch_size=16,
    #         shuffle=True)
    # X = X[:500]
    # Y = Y[:500]

    hat = DataLoader(MaleHatClass(1000), batch_size=16, shuffle=True)
    nohat = DataLoader(MaleNoHatClass(1000), batch_size=16, shuffle=True)
    
    gan = TravelGAN()
    # print(gan.G_XY)
    train(hat, nohat, gan)

if __name__ == '__main__':
    main()

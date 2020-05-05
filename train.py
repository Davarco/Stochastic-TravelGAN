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

def train(X_raw, Y_raw, gan):
    torch.autograd.set_detect_anomaly(True)

    N = min(X_raw.shape[0], Y_raw.shape[0])
    epochs = 500
    batch_size = 10
    for e in range(epochs):
        np.random.shuffle(X_raw)
        np.random.shuffle(Y_raw)

        for i in range(N//batch_size):
        # i = 0
        # for X, Y in zip(X_raw, Y_raw):
            # Train the discriminator.
            gan.D_opt.zero_grad()

            X = torch.Tensor(X_raw[i*batch_size:(i+1)*batch_size])
            Y = torch.Tensor(Y_raw[i*batch_size:(i+1)*batch_size])
            X = X.cuda()
            Y = Y.cuda()
            X = X.permute(0, 3, 1, 2) 
            Y = Y.permute(0, 3, 1, 2)

            X_gen = gan.G_YX(Y)
            Y_gen = gan.G_XY(X)

            X_logits = gan.DX(X)
            Y_logits = gan.DY(Y)
            X_gen_logits = gan.DX(X_gen)
            Y_gen_logits = gan.DY(Y_gen)

            X_dis_loss = adversarial_loss(X_logits, True)
            Y_dis_loss = adversarial_loss(Y_logits, True)
            X_gen_dis_loss = adversarial_loss(X_gen_logits, False)
            Y_gen_dis_loss = adversarial_loss(Y_gen_logits, False)
            dis_loss = X_dis_loss + X_gen_dis_loss + Y_dis_loss + Y_gen_dis_loss

            dis_loss.backward()
            gan.D_opt.step()

            # Train the generator and siamese network.
            gan.SG_opt.zero_grad()

            X = torch.Tensor(X_raw[i*batch_size:(i+1)*batch_size])
            Y = torch.Tensor(Y_raw[i*batch_size:(i+1)*batch_size])
            X = X.cuda()
            Y = Y.cuda()
            X = X.permute(0, 3, 1, 2) 
            Y = Y.permute(0, 3, 1, 2)

            X_gen = gan.G_YX(Y)
            Y_gen = gan.G_XY(X)
            X_gen_logits = gan.DX(X_gen)
            Y_gen_logits = gan.DY(Y_gen)

            SX, SY = gan.S(X, Y)
            SX_gen, SY_gen = gan.S(X_gen, Y_gen)

            X_gen_dis_loss = adversarial_loss(X_gen_logits, True)
            Y_gen_dis_loss = adversarial_loss(Y_gen_logits, True)
            vec_loss = 10 * transformation_vector_loss(SX, SX_gen)
            vec_loss += 10 * transformation_vector_loss(SY, SY_gen)
            con_loss = 10 * contrastive_loss(SX)
            con_loss += 10 * contrastive_loss(SY)

            gen_loss = X_gen_dis_loss + Y_gen_dis_loss + vec_loss
            siamese_loss = con_loss + vec_loss
            
            combined_loss = gen_loss + siamese_loss
            combined_loss.backward()

            gan.SG_opt.step()

            d = dis_loss.item()
            g = gen_loss.item()
            s = siamese_loss.item()
            t = d + g + s
            # print('Loss: (generator) {:<8.4f} (discriminator) '
            #         '{:<8.4f}'.format(combined_loss, dis_loss))
            print('Loss: (t) {:<8.4f} (d) {:<8.4f} (g) {:<8.4f} (s) {:<8.4f}'
                    .format(t, d, g, s))

            a = X_gen_dis_loss.item()
            b = Y_gen_dis_loss.item()
            t = vec_loss.item()
            c = con_loss.item()
            print('\tGen: (X_gen) {:<8.4f} (Y_gen) {:<8.4f} (TraVeL) {:<8.4f} '
                    '(Contrastive) {:<8.4f}'.format(a, b, t, c))

        # if i in check:
        if e % 100 == 0 or e == 499:
            disp_tensor_as_image(X[0])
            disp_tensor_as_image(Y[0])
            disp_tensor_as_image(X_gen[0])
            disp_tensor_as_image(Y_gen[0])
        # i += 1

def disp_tensor_as_image(X):
    img = transforms.ToPILImage()(X.cpu())
    b, g, r = img.split()
    img = Image.merge('RGB', (r, g, b))
    img.show()

def main():
    # X, Y = get_fruit_data()
    # birds = DataLoader(CifarClass('~/.torch/data', 'bird', 1000), batch_size=16,
    #         shuffle=True)
    # ships = DataLoader(CifarClass('~/.torch/data', 'ship', 1000), batch_size=16,
    #         shuffle=True)
    # X = X[:500]
    # Y = Y[:500]

    hat, nohat = get_celeb_data()
    
    gan = TravelGAN()
    train(hat, nohat, gan)

if __name__ == '__main__':
    main()

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torch.nn.functional
from torch.autograd import Variable
from torchvision import transforms
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
    epochs = 5
    batch_size = 10
    for e in range(epochs):
        np.random.shuffle(X_raw)
        np.random.shuffle(Y_raw)

        for i in range(N//batch_size):
            # Train the discriminator.
            gan.D_opt.zero_grad()

            X = torch.Tensor(X_raw[i*batch_size:(i+1)*batch_size])
            Y = torch.Tensor(Y_raw[i*batch_size:(i+1)*batch_size])
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

            dis_loss.backward(retain_graph=True)
            gan.D_opt.step()

            # Train the generator and siamese network.
            gan.SG_opt.zero_grad()

            X = torch.Tensor(X_raw[i*batch_size:(i+1)*batch_size])
            Y = torch.Tensor(Y_raw[i*batch_size:(i+1)*batch_size])
            X = X.permute(0, 3, 1, 2) 
            Y = Y.permute(0, 3, 1, 2)

            X_gen = gan.G_YX(Y)
            Y_gen = gan.G_XY(X)
            X_logits = gan.DX(X)
            Y_logits = gan.DY(Y)
            X_gen_logits = gan.DX(X_gen)
            Y_gen_logits = gan.DY(Y_gen)

            SX, SY = gan.S(X, Y)
            SX_gen, SY_gen = gan.S(X_gen, Y_gen)

            X_gen_dis_loss = adversarial_loss(X_gen_logits, True)
            Y_gen_dis_loss = adversarial_loss(Y_gen_logits, True)
            vec_loss = transformation_vector_loss(SX, SY, SX_gen, SY_gen)
            con_loss = contrastive_loss(SX, SY, SX_gen, SY_gen, 1)

            gen_loss = X_gen_dis_loss + Y_gen_dis_loss + vec_loss
            siamese_loss = con_loss + vec_loss

            gen_loss.backward(retain_graph=True)
            siamese_loss.backward()
            gan.SG_opt.step()

            d = dis_loss.item()
            g = gen_loss.item()
            s = siamese_loss.item()
            t = d + g + s
            print('Loss: (t) {:<8.4f} (d) {:<8.4f} (g) {:<8.4f} (s) {:<8.4f}'
                    .format(t, d, g, s))

            a = X_gen_dis_loss.item()
            b = Y_gen_dis_loss.item()
            t = vec_loss.item()
            c = con_loss.item()
            print('\tGen: (X_gen) {:<8.4f} (Y_gen) {:<8.4f} (TraVeL) {:<8.4f} '
                    '(Contrastive) {:<8.4f}'.format(a, b, t, c))

            if i % 10 == 0:
                disp_tensor_as_image(X[0])
                disp_tensor_as_image(Y[0])
                disp_tensor_as_image(X_gen[0])
                disp_tensor_as_image(Y_gen[0])

def disp_tensor_as_image(X):
    img = transforms.ToPILImage()(X)
    b, g, r = img.split()
    img = Image.merge('RGB', (r, g, b))
    img.show()

def main():
    X, Y = get_fruit_data()
    X = X[:500]
    Y = Y[:500]
    
    gan = TravelGAN()
    train(X, Y, gan)

if __name__ == '__main__':
    main()

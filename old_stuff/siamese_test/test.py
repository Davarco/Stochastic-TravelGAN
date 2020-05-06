import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def get_data():
    raw_data = pd.read_csv('mnist.csv').values
    split = int(len(raw_data)*0.8)

    X_train = np.matrix(raw_data[:5000, 1:], dtype=np.float32)
    Y_train = np.matrix(raw_data[:5000, :1], dtype=np.int)
    X_test = np.matrix(raw_data[split:, 1:], dtype=np.float32)
    Y_test = np.matrix(raw_data[split:, :1], dtype=np.int)

    X_train /= 255
    X_test /= 255
    # Y_train = np.squeeze(np.eye(10)[Y_train])
    # Y_test = np.squeeze(np.eye(10)[Y_test])

    return X_train, Y_train, X_test, Y_test

def main():
    X_train, Y_train, X_test, Y_test = get_data()
    X_train = torch.Tensor(X_train)
    Y_train = torch.LongTensor(Y_train)
    model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003)
    epochs = 5
    for e in range(epochs):
        for i in range(len(X_train)):
            optimizer.zero_grad()
            X = X_train[i].view(1, X_train[i].shape[0])
            Y = Y_train[i].view(Y_train[i].shape[0])
            
            output = model(X)
            loss = criterion(output, Y)
            loss.backward()
            optimizer.step()
            
            print(loss.item())

if __name__ == "__main__":
    main()

import numpy as np
from Layer import Layer
from ActivationFunctions import ReLU, Tanh, Sigmoid, Softmax
from Error import CELoss, der_CELoss, mse ,der_mse
from HelperFunctions import forward_pass, backward_pass, hot_encode

import matplotlib.pyplot as plt
import pandas as pd

train_data = pd.read_csv(r"C:\Users\pavle\OneDrive\Desktop\MNIST from schrach\csv_train.py.csv")

train_data = train_data.to_numpy()
train_data = train_data.T

X_train = train_data[1:]
Y_train = train_data[0]

X_train = X_train / 255
X_train = np.resize(X_train, (60000,784,1))
Y_train = np.resize(Y_train, (60000,1))


net = [Layer(784, 10) ,Layer(10, 10), Softmax()]
epochs = 1  
lr = 0.5


for ep in range(epochs):

    error  = 0
    acc = 0
    acc = float(acc)
    i = 0
    for x, y in zip(X_train,Y_train):
        
        pred = forward_pass(net, x)
        
        true = hot_encode(y, pred) #can also pass pred shape

        error += CELoss(true, pred)

        cat = np.argmax(pred)

        acc += (cat == np.argmax(true))

        grad = der_CELoss(y, true)

        backward_pass(net, grad, lr)

    error /= len(Y_train)
    acc /= len(Y_train)
    acc = acc * 100
    
    print(f"epoc = {ep}, error = {error} acc = {acc}")
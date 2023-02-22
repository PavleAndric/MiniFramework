import numpy as np
from Layer import Layer
from ActivationFunctions import  Tanh, Softmax
from Error import CELoss, der_CELoss
from HelperFunctions import forward_pass, backward_pass, hot_encode

import matplotlib.pyplot as plt
import pandas as pd


train_data = pd.read_csv(r"C:\Users\pavle\OneDrive\Desktop\MNIST from schrach\csv_train.py.csv")

train_data = train_data.to_numpy()
train_data = train_data.T

pre = train_data[1:]
pre_2   = train_data[0]
pre = pre / 255
X_train = pre.T 
X_train = np.expand_dims(X_train, axis = 2)
Y_train = pre_2
Y_train = np.expand_dims(Y_train ,axis = 1)

print(X_train.shape, Y_train.shape)

#net = [Layer(784, 16) ,Tanh(), Layer(16, 10), Softmax()]
#epochs =  1
#lr = 0.1


#for ep in range(epochs):

#    error  = 0.0
#    acc = 0.0
#    i = 0
#    for x, y in zip(X_train,Y_train): 
#        i = i + 1
#        pred = forward_pass(net, x)   # FORWARD
#        print(pred.shape)
#        true = hot_encode(y, pred) 
#
#        error += CELoss(true, pred)   # ERROR
#  
#        cat = np.argmax(pred) 
#        acc += (cat == np.argmax(true)) 
#
#
#        grad = der_CELoss(true, pred) 
#        if i == 5: break;


#        backward_pass(net, grad, lr)  #BACKWARD
#        
#    error /= len(Y_train)
#    acc /= len(Y_train)
#    acc = acc * 100
    
#    print(f"epoc = {ep + 1}, error = {error} acc = {acc}")
input = 800
out = 10
lejer = Layer(input, out)
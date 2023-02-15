from ActivationFunctions import Tanh
from Layer import Dense
from Error import der_mse, mse
import numpy as np

#XOR
a = np.array([[0,0],[0,1], [1,0], [1,1]])
X = np.reshape(a, (4,2,1))

b = np.array([[0], [1], [1], [0]])
Y = np.reshape(b, (4,1,1))

net = [Dense(2,3),
       Tanh(),
       Dense(3,1),
       Tanh()]
#####
ep  = 10000

lr = 0.1

for e in range(ep):

    error = 0

    for x, y in zip(X, Y):

        output = x

        for layer in net():
            output = layer.forward(output)
            print("brom")

        error += mse(y , output)
        
        grad  = der_mse(y, output)

        for layer in reversed(net):
            grad = layer.backward(grad, lr)


    error = error / len(X)
    if(e % 200  == 0 ):
        print(f" Error  is {error} , ep is {ep}")



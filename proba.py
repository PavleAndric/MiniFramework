from ActivationFunctions import Tanh
from Layer import Layer
from Error import der_mse, mse
import numpy as np
from HelperFunctions import forward_pass, backward_pass

#XOR
a = np.array([[0,0],[0,1], [1,0], [1,1]])
X = np.reshape(a, (4,2,1))

b = np.array([[0], [1], [1], [0]])
Y = np.reshape(b, (4,1,1))

net = [Layer(2,3), Tanh(), Layer(3,1), Tanh()]

ep  = 10000
lr = 0.1

for e in range(ep):

    error = 0

    for x, y in zip(X, Y):
        
        logits = forward_pass(net, x)  # FORWARD PASS

        error += mse(y , logits)       # CALCULATING THE ERROR
        
        grad  = der_mse(y, logits)     # GRAD OF LOSS
    
        backward_pass(net, grad ,lr)   # BACKPROPAGATION
        
    error = error / len(X)
    if(e % 200  == 0 ):
        print(f" Error  is {error} , ep is {e}")


from Layer import Layer
import numpy as np

class Activation(Layer):

    def __init__(self, activation, der_activation):

        self.activation = activation
        self.der_activation  = der_activation

    def forward(self,input):

        self.input  = input 
        return self.activation(input)

    def backward(self, i_grad, lr = None):
        
        return np.multiply(i_grad, self.der_activation(self.input)) #Element wise mul 
             
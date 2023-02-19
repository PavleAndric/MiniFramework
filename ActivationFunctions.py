from Activations import Activation
import numpy as np

class Tanh(Activation):

    def __init__(self):

        act = lambda input: np.tanh(input)
        der_act = lambda input: 1- np.power(np.tanh(input), 2)
        super().__init__(act, der_act)
        
class Sigmoid(Activation):

    def __init__(self):

        act  = lambda input: (1/(1 + np.exp(-input)))
        der_act = lambda input: (1/(1 + np.exp(-input))* 1- 1/(1 + np.exp(-input)))
        super().__init__(act, der_act)

class ReLU(Activation):

    def __init__(self):
        act = lambda input: np.maximum(input, 0)
        der_act = lambda input: input > 0
        super().__init__(act, der_act)

class Softmax():

    def forward(self, input):

        self.input = input - max(input)
        expo = np.exp(self.input)
        self.out = expo / np.sum(expo)
        return self.out 
    
    def backward(self,  o_grad, lr = None):
        n = np.size(self.out)
        ret = np.dot((np.identity(n) - self.out.T) * self.out, o_grad) 
        return ret 
        
        #n = np.size(self.out)
        #ret = np.dot((np.identity(n) - self.out.T) * self.out, o_grad) # shape  is 10 10
        #return ret
        
   

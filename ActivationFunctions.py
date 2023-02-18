from Activations import Activation
from Layer import Layer
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

class Softmax(Activation):

    def __init__(self):

        def act(input):
            expo = np.exp(input)
            act =  expo/ np.sum(expo , axis = 0)
            return act
        
        def der_act(input):
            pass
            
        super().__init__(act, der_act)


import Layer
import numpy as np


class Activations(Layer):
    def __init__():
        pass

    def Tanh(input):
        return np.tanh(input)
    
    def der_Tanh(intput):
        return 1 - np.power(np.tanh(input),2)
    
    def Relu(input):
        return np.maximum(0, input)
    
    def der_Relu(input):
        return input > 0 
 
import numpy as np

class Layer():

    def __init__(self, input_neurons, output_neurons):
        
        self.W = np.random.randn(output_neurons,input_neurons)
        self.B = np.random.randn(output_neurons, 1)

    def forward(self, input):

        self.X = input  #values of neurons of the current layer
        ret  = np.dot(self.W, self.X) + self.B
        return  ret #values of neurons of the next layer
    
    def backward(self ,o_grad, learning_rate):
        
        der_W = np.dot(o_grad, self.X.T)     # neuron(X) -- W --  gradient 
        der_X = np.dot(self.W.T, o_grad)        
        self.W -= der_W * learning_rate
        self.B -= o_grad * learning_rate
        return der_X 
    
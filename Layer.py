import numpy as np

class Dense():

    def __init__(self, input_neurons, output_neurons):
                
        self.W = np.random.randn(output_neurons,input_neurons)
        self.B = np.random.randn(output_neurons, 1)

    def forward(self, input):

        self.X = np.dot(self.W, input) + self.B
        return self.X
    
    def backward(self, o_grad, learning_rate):
        
        der_X = np.dot(self.W.T, o_grad)
        der_W = np.dot(o_grad, self.X.T)
        self.W -= der_W * learning_rate
        self.B -= o_grad * learning_rate
        return der_X 
    
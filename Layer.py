import numpy as np

np.random.seed(42)
class Layer():

    def __init__(self, input_neurons, output_neurons):
                
        self.W = np.random.randn(output_neurons,input_neurons)
        print(f" tezine {self.W}, shape = {self.W.shape}")
        print()
        print()
        self.B = np.random.randn(output_neurons, 1)
        print(f" pristrasnosi {self.B}, shape = {self.B.shape}")

    def forward(self, input):
        self.X = np.dot(self.W, input) + self.B
        return self.X
    
    def backward(self, o_grad, learning_rate):
        der_X = np.dot(self.W.T, o_grad)
        der_W = np.dot(o_grad, self.X.T)
        self.W -= der_W * learning_rate
        self.B -= o_grad * learning_rate
        return der_X 



print()
print()
print()
print()

layer = Layer(3,3)
output_first = np.random.randn(3,1)
print(f" output first {output_first}, o.shape = {output_first.shape} ")
x = layer.forward(output_first)
print(x)

#layer_next = Layer(3,2)

#ouptup_second = np.random.randn(3,1)
#y = layer.forward(ouptup_second)
#print(y)


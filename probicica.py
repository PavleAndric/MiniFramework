from HelperFunctions import hot_encode
from Error import CELoss
import numpy as np

def  softmax(input):
    exp = np.exp(input - np.max(input))
    return exp/ np.sum(exp , axis = 0)

def der_softmax(softmax):
    s = softmax.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)


#def der_softmax2(input):
#    n = input.size
#    der_act = np.dot((np.identity(n) - output.T) * self.output, o_grad)

sample = np.array([[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]])
#WORKS
#print(sample.shape)
#
#ret = softmax(sample)
#print(ret)
#print()
#print()
#
#print(der_softmax(ret))

sample2 = np.array([[1], [2], [3], [4]])

Y_true = [5]
Y_pred = np.array([[1],[2],[3],[4],[1],[2],[3],[4],[1],[2]])
rom = hot_encode(Y_true, Y_pred)
#print(rom)


Y_true = np.array([[1],[0],[0]])
Y_pred = np.array([[1.2], [0], [0.1234]])
sf = softmax(Y_pred)
#print(sf)

#error = CELoss(Y_true, sf)
#print(error)
input = np.random.randn(10,10)
Y_true = np.array([[1],[0]])
Y_false = np.array([[0.123], [1-0.123]])



def der_loss(Y_true, Y_pred):
    ret =((1 - Y_true) / (1 - Y_pred) - Y_true / Y_pred) / np.size(Y_true)
    return ret 

ret = der_loss(Y_true, Y_false)
print(ret.shape)
# 2x1  size 
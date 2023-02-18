from HelperFunctions import hot_encode
from Error import CELoss
from ActivationFunctions import Softmax
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
print(rom)


Y_true = np.array([[1],[0],[0]])
Y_pred = np.array([[0.7], [0.1], [0.2]])

error = CELoss(Y_true, Y_pred)
print(error)


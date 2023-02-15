import numpy as np

def mse(Y_true, Y_pred):
    return np.mean(np.power(Y_true - Y_pred), 2)

def der_mse(Y_true , Y_pred):
    ret = np.mean(Y_pred - Y_true)  
    return ret * 2  
import numpy as np

def mse(Y_true, Y_pred):

   
    return np.mean((Y_true - Y_pred) ** 2)

def der_mse(Y_true , Y_pred):
    ret = np.mean(Y_pred - Y_true)  
    return ret * 2  

def CELoss(Y_true, Y_pred):
    ret = -np.sum(Y_true * np.log(Y_pred + 1e-8))      
    return ret

def der_CELoss(Y_true, Y_pred):
    return (Y_pred - Y_true) # Tacno
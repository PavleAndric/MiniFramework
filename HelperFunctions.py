import numpy as np

def prediction(X, Y, net):
 
    for x, y in zip(X, Y):
        output = x

        for layer in net:
            output = layer.forward(output)

    print(f"predicted {output} , label = {y}")
    
def forward_pass(net, input):
    output = input
    for layer in net:
        output = layer.forward(output)
    return output

def backward_pass(net, gradient, lr):
    
    for layer in reversed(net):
        gradient = layer.backward(gradient, lr)

def hot_encode(Y_true, Y_pred):
    hot_encoded = np.zeros((Y_pred.size,1))
    hot_encoded[Y_true[0]][0] = 1
    return hot_encoded 
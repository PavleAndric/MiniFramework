### PREDS FOR XOR ONLY

def prediction(X, Y, net):
 
    for x, y in zip(X, Y):
        output = x

        for layer in net:
            output = layer.forward(output)

    print(f"predicted {output} , label = {y}")
    
def forward_pass(net, output):

    for layer in net:
        output = layer.forward(output)
    return output

def backward_pass(net, gradient, lr):

    for layer in reversed(net):
        gradient = layer.backward(gradient, lr)
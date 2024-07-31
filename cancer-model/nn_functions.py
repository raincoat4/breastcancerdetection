import numpy as np
import matplotlib.pyplot as plt
import h5py

#sigmoid activation
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

#relu activation
def relu(Z):
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache


def relu_backward(dA, cache):
    # backward propagation for a single RELU unit.
    
    Z = cache
    dZ = np.array(dA, copy=True) 

    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
   # backward propagation for a single SIGMOID unit.

    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ



def initialize_parameters_deep(layer_dims):
    
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        #set parameters to random numbers for W, and zeros for b initially
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters

def initialize_parameters_momentum(layer_dims):
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        #set both parameters to zeros
        parameters['vDw' + str(l)] = np.zeros((layer_dims[l], layer_dims[l-1]))
        parameters['vDb' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['vDw' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['vDb' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters
def linear_forward(A, W, b):
    # linear part of a layer's forward propagation.

    Z = W.dot(A) + b

    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    # activation and forward propagation for the LINEAR->ACTIVATION layer
    
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    # full forward propagation

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
        caches.append(cache)
    
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches


def compute_cost(AL, Y):
    
    m = Y.shape[1]

    # formula i found online
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    
    cost = np.squeeze(cost) #reshapes return value
    assert(cost.shape == ())
    
    return cost


def linear_backward(dZ, cache):
    # linear backward propagation
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ, A_prev.T)
    db = 1./m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    # activation + backward propagation
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    # full backward propagation basically just opposite of forward
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) 
    
    # initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    # Update parameters using gradient descent
    
    
    L = len(parameters) // 2 # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters

def update_parameters_momentum(parameters, grads, learning_rate, gradsAvg, beta):
    # Update parameters using gradient descent with momentum
    L = len(parameters) // 2

    for l in range(L):
        gradsAvg["vDw" + str(l+1)] = beta*gradsAvg["vDw" + str(l+1)] + (1-beta)*grads["dW" + str(l+1)]
        gradsAvg["vDb" + str(l+1)] = beta*gradsAvg["vDb" + str(l+1)] + (1-beta)*grads["db" + str(l+1)]
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * gradsAvg["vDw" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * gradsAvg["vDb" + str(l+1)]
        
    return parameters

def update_parameters_RMSprop(parameters, grads, learning_rate, gradsAvg, beta):
    #Update parameters using gradient descent with rmsprop
    L = len(parameters) // 2

    for l in range(L):
        gradsAvg["vDw" + str(l+1)] = beta*gradsAvg["vDw" + str(l+1)] + (1-beta)*grads["dW" + str(l+1)]**2
        gradsAvg["vDb" + str(l+1)] = beta*gradsAvg["vDb" + str(l+1)] + (1-beta)*grads["db" + str(l+1)]**2
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]/np.sqrt(gradsAvg["vDw" + str(l+1)] + 10**-8)
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]/np.sqrt(gradsAvg["vDb" + str(l+1)] + 10**-8)
        
    return parameters

def predict(X, y, parameters):
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # forward propagation
    probas, caches = L_model_forward(X, parameters)

    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p

# -*- coding: utf-8 -*-

        # Video 1: Intro + Neuron code

# =============================================================================
'''
Key words
## 1. T = Transpose

Neural network process
# 1. weights * input values for all elements (dot product)
    z1[0] = w1[0,0] * x1[0] + w1[0,1] * x1[1] + ... + w1[0,199] * x1[199]
    = z1 = [np.dot(x1, w1[0].T) + b1[0], np.dot(x1, w1[1].T) + b1[1]]
    = z1 = np.dot(x1, w1.T) + b1
# 2. y1 = max(0, dot product output) -> Do for all layers
    y1 = ReLU(z1) = np.maximum(0, z1 )
        ReLU = rectified linear 
    = y1 = np.maximum(0, np.dot(w1.T) + b1)
    y2 = np.maximum(0, np.dot(x2, w2.T) + b2)
# 3. y_hat = Softmax activation function with bias function
    y_hat1 = softmax(z3)[0] = np.exp(z3) / np.sum(np.exp(z3), axis=1, keepdims=T)
    y_hat2 = softmax(z3)[1] = np.exp(z3) / np.sum(np.exp(z3), axis=1, keepdims=T)
    z3 = np.dot(x3, w3.T) + b3
# 4. Calculate loss = Negative log loss 
    loss = -np.log(np.sum(y * y_hat))
    

Final function
# loss = -np.log(np.sum(y * \
    np.exp(np.dot(np.maximum(0, np.dot(np.maximum (0, np.dot(X, w1.T) + \
                                                   b1), w2.T) + b2), w3.T) + b3) / \
    np.sum(np.exp(np.dot(np.maximum(0, np.dot(np.maximum(0, np.dot(X, w1.T) + b1), w2.T) + \
    b2), w3.T) + b3), axis=1), keepdims=True)
  ))


BKG in neural network 
    # Input (2 nodes) -> Hidden layer 1 (4 nodes) -> Hidden layer 2 (4 nodes) -> Output layer (2 nodes)
    # EG check if somethings a failure or not a failure (^ value = our prediction)
    # EG input pictures of cats or dogs and the network predicts which one it is
    # Tuning the model = Adjusting weights and bias values to make more accurate predictions
'''

# Make a neuron
    # Neurons always have connections to previous and next neurons
    # Previous neuron's output = Next neuron's input
    
inputs = [1, 2, 3] # random outputs from 3 neurons of previous layer
weights = [0.2, 0.8, -0.5] # weights always accompany a neuron 
bias = 2 # every 1 unique neuron has 1 unique bias

output = inputs[0] * weights[0] + inputs[1] * weights[1] + \
    inputs[2] * weights[2] + bias

print(output) # prints 2.3

# =============================================================================

        # Video 2: Coding a layer

# =============================================================================
inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1]
bias = 2

output = inputs[0] * weights[0] + inputs[1] * weights[1] + \
    inputs[2] * weights[2] + inputs[3] * weights[3] + bias
    
print(output) # 4.8

# Coding 3 neurons (current) with 4 inputs (previous)
    # EG finding inputs of hidden layer with 3 neurons when hidden layer has 4 outputs
inputs = [1, 2, 3, 2.5]

weights1 = [0.2, 0.8, -0.5, 1] # Have 3 sets of weights each with 4 values
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5

output = [inputs[0] * weights1[0] + inputs[1] * weights1[1] + inputs[2] * weights1[2] + inputs[3] * weights1[3] + bias1,
          inputs[0] * weights2[0] + inputs[1] * weights2[1] + inputs[2] * weights2[2] + inputs[3] * weights2[3] + bias2, 
          inputs[0] * weights3[0] + inputs[1] * weights3[1] + inputs[2] * weights3[2] + inputs[3] * weights3[3] + bias3]
    # Made into list with "[,]"

print(output) # 4.8, 1.21, 2.385

# =============================================================================

        # Video 3: Dot product

# =============================================================================
'''
# 'zip' = combines 2 lists into list of lists element wise
# Batches = Allow us to calculate things in parallel (More simulations done simultaneously)
# Can be taxing = Do it on GPUs rather than CPUs
# dot products can be visualised with matrices
    # sum of First row from input matrix * first column of weight matrix
        # Continue to all to make matrix product
# inputs and weight matrix = Same shape (4 by 3) = Cannot do dot product
    # i.e., need to transpose the weight matrix (switch rows + columns)
        # 'list' has no attribute 'T' = Need to make into array
'''            
# method 1: long version (where shape doesn't matter)
inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2,3,0.5]

layer_outputs = [] # output of current layer
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0 # output of given neuron
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input * weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)
    
print(layer_outputs) #prints [4.8, 1.21, 2.385]
        
# =============================================================================

        # Video 4: Batches, Layers, Objects

# =============================================================================
# method 2: introducing different layers 
import numpy as np

inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2,3,0.5]
                     
weights2 = [[0.1, -0.14, 0.5],
           [-0.5, 0.12, -0.33],
           [-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5]


# layer 1 outputs becomes layer 2 inputs
layer1_outputs = np.dot(inputs, np.array(weights).T) + biases 
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2 
print(layer1_outputs, layer2_outputs)
    # [[ 4.8    1.21   2.385]
     # [ 8.9   -1.81   0.2  ]
     # [ 1.41   1.051  0.026]] 
     # [[ 0.5031  -1.04185 -2.03875]
     # [ 0.2434  -2.7332  -5.7633 ]
     # [-0.99314  1.41254 -0.35655]]

# method 3: Dense Layers (Forward method)
'''
# Using class method to provide a cleaner solution
# Currently, increasing number of layers just increases the values, but that does not give prediction value at all
# Want to confine the numbers from 1 to -1, can show binary or other numerical value
    # i.e., normalise and scale the inputs and weights(-0.1 to 0.1)
    # Want to initialise biases =/= 0 (Could get a dead network of only 0s)

'''
    
import numpy as np

np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]] # Standard ML 'X' denotes input feature set


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# 1. Initialise weights -> random values of range -0.1 to 0.1
    # tighter the range the better for calulations
# 2. Initialise biases as 0
    # need to be careful of zeros network
# 3. random seed = weights
# 4. randn() = looks at the shapes 
    # randn() = Gaussian distribution bound around 0
    # Want to * 0.1 to make the weight values small
# 5. np.zeros() = Looks at the shape, makes a matrix of 0s with shape (1, n_neurons)
    # need to pass the actual shape as the parameter = double brackets (tuple)



layer1 = Layer_Dense(4,5) #size of inputs, number of neurons we want, make it up
layer2 = Layer_Dense(5,2) #size of input must be the same of n_neurons in layer1

layer1.forward(X)
# print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)
    
    
# =============================================================================

    #  Video 5: Hidden Layer Activation Functions (Introducing activation functions)
    
# =============================================================================
''' 
Tyes of activation functions:
1. Step function
   # input = x, output = y
2. Recified Linear (ReLU)
3. Sigmoid
4. Linear
5. Softmax

How do step functions work:
 # Each neurons in hidden and outer layers will have activation function
 # Only happens after inputs * weights + bias = becomes input for activation function
 1. Step function (EG if x>0 y=1, if x<=0 y=0)
 2. Sigmoid function (EG y = 1 / (1 + e^(-x)))
     # May be more reliable than step function due to the gradual curve of its movement
     # More 'granular' output of the function = makes it easier to optimise the network
     # Issue: Vanishing gradient problem, slightly more complex = slower
 3. ReLU (EG if x>0 y=x, if x<=0 y=0)
     # output can be granular
     # Granular = Can help optimise network
     # simple calculation = fast
     # Most popular activation function

Why do we use (non-linear) activation functions
# If we dont have activation function, weights and biases would have a linear relationship
    # Can only fit linear data (limited + inaccurate)
    # Cannot fit linear function to non-linear relationships
# Using non-linear activation fucntion like ReLU can more accurately fit to data
# Therefore:
    # Activation functions introduces non-linearity 
    # Helps with more complex decision making (such as classifications)
    # Controls output of neurons (EG keep values between 0-1, softmax for multi-class classifications)
    # Affects gradients used in optimisation algorithms like gradient descent
        # Helps resolve issues like vanishing gradient 
    # Helps network learn more abstract features at higher layers
        # useful in image and speech recognition

Why does ReLU function work
# Change bias = change y-intercept 
# Change weight +ve to -ve = flips the graph on y-axis = Input deactivates 
# Change bias on 2nd neuron = graph goes up or down
    # y = f(wx + b)
        # x > -(b/w) = Produce linear output, slope of 'w'
        # x <= -(b/w) = Produce output of 0, slope of 0 
'''
# Introducing ReLU into the code

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []

## method 1
for i in inputs:
    if i > 0:
        output.append(i)
    elif i <= 0:
        output.append(0)
        
print(output)

## method 2
for i in inputs:
    output.append(max(0,i))
        
print(output)


# writing the ReLU object with forward and backward method
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

X, y = spiral_data (100,3) 

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        
        
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        
layer1 = Layer_Dense(2,5) # 1st layer after inputs
    # 4 = #inputs, 5= #neurons
activation1 = Activation_ReLU() # Makes all inputs go thru ReLU function

layer1.forward(X)
activation1.forward(layer1.output)
print(activation1.output)

# =============================================================================

        # Video 6: Softmax activation
# =============================================================================
'''
# Softmax activation = Exponentiation + Normalisation of inputs 
    # = Affects output layer for classification style analysis
    # Takes vector of raw inputs -> Converts to probabilities that sums to 1
    # sigma(zi) = e^(zi) / (Sum(j) * e^(zi))
        # zi = i-th element of input vector z
        # e = natural log
        # denominator = sum of exponents of all values in vector
# Softmax characteristics
    # Output is probability distribution = output between 0-1
    # Normalised = output probability sums to 1
        # Good for modeling categorical probabilities
    # Sensitive = Amplifies differences between input values
        # larger logits = higher probabilities
        # Smaller logits = probilities closer to 0
# Uses of softmax
    # Multi-class classification
        # Each output node corresponds to a class
        # Function interprets raw output as probabilities
    # Loss function
        # If paired with categorical cross-entropy loss function = compares predicted 
            #probabilities with true class labels
# Use exponents to avoid the issues with negative values from ReLU function
    # Need to normalise the values to create a probability distribution
        # i.e., 2/3 and 1/3 between 2 nodes
'''

# numpy implementation for exponents with batches
import numpy as np
layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_outputs)

norm_values = exp_values / np.sum(layer_outputs, axis=1, keepdims = True) # Sum of rows (by batches) also keep same dimensions

print(norm_values)

'''
An issue exponentials have is that the output value increases dramatically with
incremetal increase in input. Need a method to prevent overflow

Overflow prevention (v)= u - max(u)
i.e., subtract all values from output layer prior to exponentiation to the 
largest value from that layer. Thus, the largest value will end with a value of 0,
and the rest will be smaller than 0 (If -inf < x < 0 , then 0 > y > 1)
'''

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

X,y = spiral_data(samples = 100, classes=3)

dense1 = Layer_Dense(2,3) # Spiral data only gives x,y coordinates = input =2)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)
# =============================================================================


        # Appendix
# =============================================================================
# Spiral dataset  

#https://cs231n.github.io/neural-networks-case-study/

import numpy as np
np.random.seed(0)

def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

import matplotlib.pyplot as plt

X,y = spiral_data(100,3)

plt.scatter(X[:,0], X[:,1])
plt.show

plt.scatter(X[:,0], X[:,1], c=y, cmap="brg")
plt.show

# =============================================================================

import numpy as np
import math
from statistics import mean

class NeuralNetwork:
    
    def __init__(self, inputSize = 4, weights:np.ndarray = None, bias = 0):
        self.inputSize = inputSize
        
        if type(weights) != np.ndarray:
            self.weights = np.zeros(self.inputSize)
        elif weights.shape == np.zeros((self.inputSize)).shape:
            self.weights = weights.copy()
        else:
            raise Exception(f"Shape must be ({self.inputSize}), not {weights.shape}")
        
        self.bias = bias
    
    @staticmethod
    def sigmoid(value):
        return 1.0 / (1 + math.e ** (-value))

    def predict(self, x):
        
        if type(x) == tuple:
            x = np.asarray(x)
        
        if x.shape != self.weights.shape:
            raise Exception(f"Shape must be ({self.inputSize}), not {x.shape}")

        return NeuralNetwork.sigmoid(np.dot(x, self.weights) + self.bias)
    
class Training:
    
    def __init__(self):
        pass
    
    @staticmethod
    def MSE(data, neuralNetwork:NeuralNetwork, expectedValues):
        errors = [(expectedValues[i] - neuralNetwork.predict(paramSet)) ** 2 for i, paramSet in enumerate(data)]
        
        return mean(errors)
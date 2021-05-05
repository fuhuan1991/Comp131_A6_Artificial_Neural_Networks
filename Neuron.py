import math

class Neuron:

    def __init__(self, alpha, W):
        self.alpha = alpha
        self.W = W.copy()

    def activationFunction(self, x):
        return 1 / (1 + math.e ** (-x))

    # The derivative of the activation function
    def derivative(self, x):
        return math.e ** x / (math.e ** x + 1) ** 2

    # get the weighted sum of input(potential) of this neuron
    def getPotential(self, X):
        x1 = X[0]
        x2 = X[1]
        x3 = X[2]
        x4 = X[3]
        w0 = self.W[0]
        w1 = self.W[1]
        w2 = self.W[2]
        w3 = self.W[3]
        w4 = self.W[4]
        return w0 + w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4

    def run(self, X):
        potential = self.getPotential(X)
        return self.activationFunction(potential)

    



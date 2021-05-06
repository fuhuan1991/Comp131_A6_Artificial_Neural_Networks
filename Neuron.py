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
        w0 = self.W[0]
        length = len(X)
        sum = w0
        for i in range(length):
            sum = sum + X[i] * self.W[i + 1]

        return sum

    def run(self, X):
        potential = self.getPotential(X)
        return self.activationFunction(potential)

    



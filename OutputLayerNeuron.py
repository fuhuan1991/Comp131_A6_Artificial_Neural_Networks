from Neuron import Neuron

class OutputLayerNeuron(Neuron):

    # The calculation of the error, 
    # it is based on the difference between desired and actual output 
    # X: input vector
    # y: desired output
    # h_wx: actual output
    def getErrorValue(self, X, y, h_wx):
        p = self.getPotential(X)
        return self.derivative(p) * (y - h_wx)

    # Train this neuron with a correct output
    # W will be updated according to the difference between desired output and the actual output
    # X: input vector
    # y: desired output
    # h_wx: actual output 
    def train(self, X, y, h_wx):
        errorValue = self.getErrorValue(X, y, h_wx)
        length = len(X)
        self.W[0] = self.W[0] + self.alpha * errorValue
        for i in range(length):
            self.W[i + 1] = self.W[i + 1] + self.alpha * errorValue * X[i]
        return self.W



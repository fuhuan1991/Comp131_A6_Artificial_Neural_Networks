from Neuron import Neuron

class HiddenLayerNeuron(Neuron):

    # The calculation of the error, 
    # it is based on the error values of the next layer
    # X: input vector
    # Err_j: errorValues of next layer, an array
    # W_ij: an array of weights between this neuron and the neurons in the next layer
    def getErrorValue(self, X, Err_j, W_ij):
        errSum = 0
        length = len(W_ij)
        for j in range(length):
            errSum = errSum + W_ij[j] * Err_j[j]
        p = self.getPotential(X)
        return self.derivative(p) * errSum

    # Train this neuron 
    # W will be updated according to the error values from the next layer
    # X: input vector
    # Err_j: errorValues of next layer, an array
    # W_ij: an array of weights between this neuron and the neurons in the next layer
    def train(self, X, Err_j, W_ij):
        # print("old W = " + str(self.W))
        errorValue = self.getErrorValue(X, Err_j, W_ij)
        length = len(X)
        self.W[0] = self.W[0] + self.alpha * errorValue
        for i in range(length):
            self.W[i + 1] = self.W[i + 1] + self.alpha * errorValue * X[i]
        
        # print("X = " + str(X))
        # print(errorValue)
        # print("new W = " + str(self.W))
        # print("      ")
        return self.W


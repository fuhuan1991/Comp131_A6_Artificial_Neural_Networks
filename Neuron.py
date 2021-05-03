

class Neuron:

    def __init__(self, alpha, W):
        self.alpha = alpha
        self.W = W 

    def ActivationFunction(self, input):
        if input > 0:
            return 1
        else:
            return 0

    def run(self, X):
        x1 = X[0]
        x2 = X[1]
        x3 = X[2]
        x4 = X[3]
        w0 = self.W[0]
        w1 = self.W[1]
        w2 = self.W[2]
        w3 = self.W[3]
        w4 = self.W[4]
        v = w0 + w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4
        return self.ActivationFunction(v)

    def train(self, X, y, h_wx):
        # print([y, h_wx])
        if y == 1 and h_wx == 0:
            self.W[0] = self.W[0] + self.alpha
            for i in range(4):
                self.W[i + 1] = self.W[i + 1] + self.alpha * X[i]

            # print("y: 1 h_wx: 0 new W = " + str(self.W))

        if y == 0 and h_wx == 1:
            self.W[0] = self.W[0] - self.alpha
            for i in range(4):
                self.W[i + 1] = self.W[i + 1] - self.alpha * X[i]

            # print("y: 0 h_wx: 1 new W = " + str(self.W))



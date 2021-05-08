import random


def getRandomWeight(W):
    weights = W.copy()
    length = len(W)
    for i in range(length):
        weights[i] += (random.random() - 0.5) / 5
    return weights

W = [0, 0, 0, 0, 0]

from Perceptron import Perceptron
import random

f = open("ANN - Iris data.txt", "r")
trainData = []
testData = []
len = 150
testDataAmount = 10  # no more than 50

for i in range(3):
    for j in range(50):
        line = f.readline()
        X = []
        arr = line.split(',')
        X.append(float(arr[0]))
        X.append(float(arr[1]))
        X.append(float(arr[2]))
        X.append(float(arr[3]))
        X.append(arr[4][:-1])
        if j < testDataAmount:
            testData.append(X)
        else:
            trainData.append(X)

f.close()

random.shuffle(trainData)
random.shuffle(testData)

# for d in trainData:
#     print(d)
# for d in testData:
#     print(d)

alpha = 0.1
W = [0, 0, 0, 0, 0]


p = Perceptron(alpha, W)

for i in range(150):
    dataPoint = data[i]
    h_wx = p.run(dataPoint)
    y = 0
    if dataPoint[4] == "Iris-setosa":
        y = 1
    else:
        y = 0
    
    p.train(dataPoint, y, h_wx)
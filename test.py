from Neuron import Neuron
import random

f = open("ANN - Iris data.txt", "r")
trainData = [] # data for training
testData = [] # data for testing
len = 150
testDataAmount = 10  # The number of data points for testing in each type of iris, no more than 50
alpha = 0.1 # learning rate
W = [0, 0.25, 0.25, 0.25, 0.25] # Initial weight

# Devide test data into 3 parts, each contains only one type of iris
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
        # A portion of data would be used for testing whild others would be used for training
        if j < testDataAmount:
            testData.append(X)
        else:
            trainData.append(X)

f.close()

# shuffle the data points.
random.shuffle(trainData)
random.shuffle(testData)

# for d in trainData:
#     print(d)
# for d in testData:
#     print(d)

# Generate Neurons 
p = Neuron(alpha, W)

# Train the ANN
for dataPoint in trainData:
    h_wx = p.run(dataPoint)
    y = 0
    if dataPoint[4] == "Iris-setosa":
        y = 1
    else:
        y = 0
    
    p.train(dataPoint, y, h_wx)

# Test the ANN
correctCounter = 0
totalTestCounter = testDataAmount * 3
for dataPoint in testData:
    h_wx = p.run(dataPoint)
    y = 0
    if dataPoint[4] == "Iris-setosa":
        y = 1
    else:
        y = 0
    if (y == h_wx):
        correctCounter += 1

print(str(correctCounter) + " out of " + str(totalTestCounter) + " test cases are correct.")
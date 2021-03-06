from OutputLayerNeuron import OutputLayerNeuron
from HiddenLayerNeuron import HiddenLayerNeuron
import random
from util import getRandomWeight

# In this file, the ANN has a single-layer structure
############################# Setup constants
print("---In this file, the ANN has a multi-layer structure---")
f = open("ANN - Iris data.txt", "r")
trainData = [] # data for training
testData = [] # data for testing
len = 150
testDataAmount = 10  # The number of data points for testing in each type of iris, no more than 50
alpha = 0.4 # learning rate
W = [0, 0.1, 0.1, 0.1, 0.1] # Initial weight

############################# Prepare data

# Devide test data into 3 parts, each contains only one type of iris
for i in range(3):
    for j in range(50):
        line = f.readline()
        dataPoint = []
        arr = line.split(',')
        dataPoint.append(float(arr[0]))
        dataPoint.append(float(arr[1]))
        dataPoint.append(float(arr[2]))
        dataPoint.append(float(arr[3]))
        dataPoint.append(arr[4][:-1])
        # A portion of data would be used for testing whild others would be used for training
        if j < testDataAmount:
            testData.append(dataPoint)
        else:
            trainData.append(dataPoint)

f.close()

# shuffle the data points.
random.shuffle(trainData)


############################# Build the ANN

# Generate 3 neurons for the output layer, each for a type of iris
o1 = OutputLayerNeuron(alpha, getRandomWeight(W)) # Iris-setosa
o2 = OutputLayerNeuron(alpha, getRandomWeight(W)) # Iris-versicolor
o3 = OutputLayerNeuron(alpha, getRandomWeight(W)) # Iris-virginica

# Generate 4 neurons for the hidden layer
# In this case, only one hidden layer
n1 = HiddenLayerNeuron(alpha, getRandomWeight(W))
n2 = HiddenLayerNeuron(alpha, getRandomWeight(W))
n3 = HiddenLayerNeuron(alpha, getRandomWeight(W))
n4 = HiddenLayerNeuron(alpha, getRandomWeight(W))


############################# Train the ANN

for dataPoint in trainData:
    X = [dataPoint[0], dataPoint[1], dataPoint[2], dataPoint[3]]

    # generate output
    a_1 = n1.run(X)
    a_2 = n2.run(X)
    a_3 = n3.run(X)
    a_4 = n4.run(X)

    hiddenLayerOutput = [a_1, a_2, a_3, a_4]
    output_1 = o1.run(hiddenLayerOutput) # Iris-setosa
    output_2 = o2.run(hiddenLayerOutput) # Iris-versicolor
    output_3 = o3.run(hiddenLayerOutput) # Iris-virginica

    # train output layer neurons
    desiredOutput = []
    W1 = []
    W2 = []
    W3 = []

    # generate disired output
    if (dataPoint[4] == "Iris-setosa"): 
        desiredOutput = [1, 0, 0]
    if (dataPoint[4] == "Iris-versicolor"): 
        desiredOutput = [0, 1, 0]
    if (dataPoint[4] == "Iris-virginica"): 
        desiredOutput = [0, 0, 1]
    W1 = o1.train(hiddenLayerOutput, desiredOutput[0], output_1)
    W2 = o2.train(hiddenLayerOutput, desiredOutput[1], output_2)
    W3 = o3.train(hiddenLayerOutput, desiredOutput[2], output_3)

    # train hidden layer neurons
    W_n1 = [W1[1], W2[1], W3[1]]
    W_n2 = [W1[2], W2[2], W3[2]]
    W_n3 = [W1[3], W2[3], W3[3]]
    W_n4 = [W1[4], W2[4], W3[4]]

    err_01 = o1.getErrorValue(hiddenLayerOutput, desiredOutput[0], output_1) 
    err_02 = o2.getErrorValue(hiddenLayerOutput, desiredOutput[1], output_2) 
    err_03 = o3.getErrorValue(hiddenLayerOutput, desiredOutput[2], output_3)
    Err = [err_01, err_02, err_03]

    n1.train(X, Err, W_n1)
    n2.train(X, Err, W_n2)
    n3.train(X, Err, W_n3)


############################# Test the ANN

correctCounter = 0
totalTestCounter = testDataAmount * 3
for dataPoint in testData:

    X = [dataPoint[0], dataPoint[1], dataPoint[2], dataPoint[3]]

    desiredOutput = []

    if (dataPoint[4] == "Iris-setosa"): 
        desiredOutput = [1, 0, 0]
    if (dataPoint[4] == "Iris-versicolor"): 
        desiredOutput = [0, 1, 0]
    if (dataPoint[4] == "Iris-virginica"): 
        desiredOutput = [0, 0, 1]
    
    # generate output
    a_1 = n1.run(X)
    a_2 = n2.run(X)
    a_3 = n3.run(X)
    a_4 = n4.run(X)

    hiddenLayerOutput = [a_1, a_2, a_3, a_4]
    output_1 = o1.run(hiddenLayerOutput) # Iris-setosa
    output_2 = o2.run(hiddenLayerOutput) # Iris-versicolor
    output_3 = o3.run(hiddenLayerOutput) # Iris-virginica

    actualOutput = [output_1, output_2, output_3]
    print("desiredOutput: " + str(desiredOutput) + " actualOutput: " + str(actualOutput))
    for i in range(3):
        if actualOutput[i] > 0.5:
            actualOutput[i] = 1
        else:
            actualOutput[i] = 0

    if desiredOutput[0] == actualOutput[0] and desiredOutput[1] == actualOutput[1] and desiredOutput[2] == actualOutput[2]:
        correctCounter += 1

    # print("desiredOutput: " + str(desiredOutput) + " actualOutput: " + str(actualOutput))

print(str(correctCounter) + " out of " + str(totalTestCounter) + " test cases are correct.")
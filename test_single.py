from OutputLayerNeuron import OutputLayerNeuron
from HiddenLayerNeuron import HiddenLayerNeuron
import random
from util import getRandomWeight
# In this file, the ANN has a single-layer structure
############################# Setup constants
print("---In this file, the ANN has a single-layer structure---")
f = open("ANN - Iris data.txt", "r")
trainData = [] # data for training
testData = [] # data for testing
len = 150
testDataAmount = 10  # The number of data points for testing in each type of iris, no more than 50
alpha = 0.4 # learning rate
W = [0, 0, 0, 0, 0] # Initial weight

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
# random.shuffle(testData)


############################# Build the ANN

# Generate 3 neurons for the output layer, each for a type of iris
o1 = OutputLayerNeuron(alpha, getRandomWeight(W)) # Iris-setosa
o2 = OutputLayerNeuron(alpha, getRandomWeight(W)) # Iris-versicolor
o3 = OutputLayerNeuron(alpha, getRandomWeight(W)) # Iris-virginica


############################# Train the ANN

for dataPoint in trainData:
    X = [dataPoint[0], dataPoint[1], dataPoint[2], dataPoint[3]]
    
    # generate ouput from input data
    output_1 = o1.run(X) # Iris-setosa
    output_2 = o2.run(X) # Iris-versicolor
    output_3 = o3.run(X) # Iris-virginica

    # train output layer neurons, with corect answer
    desiredOutput = []
    W1 = []
    W2 = []
    W3 = []
    if (dataPoint[4] == "Iris-setosa"): 
        desiredOutput = [1, 0, 0]
    if (dataPoint[4] == "Iris-versicolor"): 
        desiredOutput = [0, 1, 0]
    if (dataPoint[4] == "Iris-virginica"): 
        desiredOutput = [0, 0, 1]
    W1 = o1.train(X, desiredOutput[0], output_1)
    W2 = o2.train(X, desiredOutput[1], output_2)
    W3 = o3.train(X, desiredOutput[2], output_3)

    # print(desiredOutput)
    # print([output_1, output_2, output_3])

    # print(o1.W)
    # print(o2.W)
    # print(o3.W)

    # print("--------")
    

############################# Test the ANN

correctCounter = 0
totalTestCounter = testDataAmount * 3
for dataPoint in testData:
    X = [dataPoint[0], dataPoint[1], dataPoint[2], dataPoint[3]]
    desiredOutput = 0

    # The ANN can only tell the difference between Iris-setosa and the other 2 types
    if (dataPoint[4] == "Iris-setosa"): 
        desiredOutput = 1
    if (dataPoint[4] == "Iris-versicolor"): 
        desiredOutput = 0
    if (dataPoint[4] == "Iris-virginica"): 
        desiredOutput = 0
    
    # output_1 is the related output neuron of the type Iris-setosa
    output_1 = o1.run(X) # Iris-setosa

    actualOutput = output_1
    print("desiredOutput: " + str(desiredOutput) + "    actualOutput: " + str(actualOutput))
    
    
    if actualOutput > 0.5:
        actualOutput = 1
        print("the target belongs to Iris-setosa")
    else: 
        actualOutput = 0
        print("the target belongs the other 2 types")

    if desiredOutput == actualOutput:
        correctCounter += 1

    # print("desiredOutput: " + str(desiredOutput) + " actualOutput: " + str(actualOutput))

print(str(correctCounter) + " out of " + str(totalTestCounter) + " test cases are correct.")
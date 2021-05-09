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
alpha = 2 # learning rate
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
# random.shuffle(testData)

############################# Build the ANN

# Generate 3 neurons for the output layer, each for a type of iris
# o1 = OutputLayerNeuron(alpha, getRandomWeight([0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])) # Iris-setosa
o2 = OutputLayerNeuron(alpha, getRandomWeight([0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])) # Iris-versicolor
# o3 = OutputLayerNeuron(alpha, getRandomWeight(W)) # Iris-virginica

# Generate 4 neurons for the hidden layer
# In this case, only one hidden layer
n1 = HiddenLayerNeuron(alpha, getRandomWeight(W))
n2 = HiddenLayerNeuron(alpha, getRandomWeight(W))
n3 = HiddenLayerNeuron(alpha, getRandomWeight(W))
n4 = HiddenLayerNeuron(alpha, getRandomWeight(W))
n5 = HiddenLayerNeuron(alpha, getRandomWeight(W))
n6 = HiddenLayerNeuron(alpha, getRandomWeight(W))

m1 = HiddenLayerNeuron(alpha, getRandomWeight([0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
m2 = HiddenLayerNeuron(alpha, getRandomWeight([0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
m3 = HiddenLayerNeuron(alpha, getRandomWeight([0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
m4 = HiddenLayerNeuron(alpha, getRandomWeight([0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
m5 = HiddenLayerNeuron(alpha, getRandomWeight([0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
m6 = HiddenLayerNeuron(alpha, getRandomWeight([0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))



############################# Train the ANN

for dataPoint in trainData:
    X = [dataPoint[0], dataPoint[1], dataPoint[2], dataPoint[3]]

    # generate output
    a_1 = n1.run(X)
    a_2 = n2.run(X)
    a_3 = n3.run(X)
    a_4 = n4.run(X)
    a_5 = n5.run(X)
    a_6 = n6.run(X)

    hiddenLayerOutput_1 = [a_1, a_2, a_3, a_4, a_5, a_6]

    b_1 = m1.run(hiddenLayerOutput_1)
    b_2 = m2.run(hiddenLayerOutput_1)
    b_3 = m3.run(hiddenLayerOutput_1)
    b_4 = m4.run(hiddenLayerOutput_1)
    b_5 = m5.run(hiddenLayerOutput_1)
    b_6 = m6.run(hiddenLayerOutput_1)

    hiddenLayerOutput_2 = [b_1, b_2, b_3, b_4, b_5, b_6]

    # output_1 = o1.run(hiddenLayerOutput) # Iris-setosa
    output_2 = o2.run(hiddenLayerOutput_2) # Iris-versicolor
    # output_3 = o3.run(hiddenLayerOutput) # Iris-virginica

    # train output layer neurons
    desiredOutput = []
    # W1 = []
    W2 = []
    # W3 = []
    
    # generate disired output
    if (dataPoint[4] == "Iris-setosa"): 
        desiredOutput = [1, 0, 0]
    if (dataPoint[4] == "Iris-versicolor"): 
        desiredOutput = [0, 1, 0]
    if (dataPoint[4] == "Iris-virginica"): 
        desiredOutput = [0, 0, 1]
    # W1 = o1.train(hiddenLayerOutput, desiredOutput[0], output_1)
    W_output_2 = o2.train(hiddenLayerOutput_2, desiredOutput[1], output_2)
    # W3 = o3.train(hiddenLayerOutput, desiredOutput[2], output_3)

    W_m_o_1 = [W_output_2[1]]
    W_m_o_2 = [W_output_2[2]]
    W_m_o_3 = [W_output_2[3]]
    W_m_o_4 = [W_output_2[4]]
    W_m_o_5 = [W_output_2[5]]
    W_m_o_6 = [W_output_2[6]]

    # err_01 = o1.getErrorValue(hiddenLayerOutput, desiredOutput[0], output_1) 
    err_02 = o2.getErrorValue(hiddenLayerOutput_2, desiredOutput[1], output_2) 
    # err_03 = o3.getErrorValue(hiddenLayerOutput, desiredOutput[2], output_3)
    # Err = [err_01, err_02, err_03]
    Err_output = [err_02]

    W_m_1 = m1.train(hiddenLayerOutput_1, Err_output, W_m_o_1)
    W_m_2 = m2.train(hiddenLayerOutput_1, Err_output, W_m_o_2)
    W_m_3 = m3.train(hiddenLayerOutput_1, Err_output, W_m_o_3)
    W_m_4 = m4.train(hiddenLayerOutput_1, Err_output, W_m_o_4)
    W_m_5 = m5.train(hiddenLayerOutput_1, Err_output, W_m_o_5)
    W_m_6 = m6.train(hiddenLayerOutput_1, Err_output, W_m_o_6)

    W_n_m_1 = [W_m_1[1], W_m_2[1], W_m_3[1], W_m_4[1], W_m_5[1], W_m_6[1]]
    W_n_m_2 = [W_m_1[2], W_m_2[2], W_m_3[2], W_m_4[2], W_m_5[2], W_m_6[2]]
    W_n_m_3 = [W_m_1[3], W_m_2[3], W_m_3[3], W_m_4[3], W_m_5[3], W_m_6[3]]
    W_n_m_4 = [W_m_1[4], W_m_2[4], W_m_3[4], W_m_4[4], W_m_5[4], W_m_6[4]]
    W_n_m_5 = [W_m_1[5], W_m_2[5], W_m_3[5], W_m_4[5], W_m_5[5], W_m_6[5]]
    W_n_m_6 = [W_m_1[6], W_m_2[6], W_m_3[6], W_m_4[6], W_m_5[6], W_m_6[6]]

    err_m_1 = m1.getErrorValue(hiddenLayerOutput_1, Err_output, W_m_o_1)
    err_m_2 = m2.getErrorValue(hiddenLayerOutput_1, Err_output, W_m_o_1)
    err_m_3 = m3.getErrorValue(hiddenLayerOutput_1, Err_output, W_m_o_1)
    err_m_4 = m4.getErrorValue(hiddenLayerOutput_1, Err_output, W_m_o_1)
    err_m_5 = m5.getErrorValue(hiddenLayerOutput_1, Err_output, W_m_o_1)
    err_m_6 = m6.getErrorValue(hiddenLayerOutput_1, Err_output, W_m_o_1)

    Err_output = [err_m_1, err_m_2, err_m_3, err_m_4, err_m_5, err_m_6]

    n1.train(X, Err_output, W_n_m_1)
    n2.train(X, Err_output, W_n_m_2)
    n3.train(X, Err_output, W_n_m_3)
    n4.train(X, Err_output, W_n_m_4)
    n5.train(X, Err_output, W_n_m_5)
    n6.train(X, Err_output, W_n_m_6)

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
    a_5 = n5.run(X)
    a_6 = n6.run(X)

    hiddenLayerOutput_1 = [a_1, a_2, a_3, a_4, a_5, a_6]

    b_1 = m1.run(hiddenLayerOutput_1)
    b_2 = m2.run(hiddenLayerOutput_1)
    b_3 = m3.run(hiddenLayerOutput_1)
    b_4 = m4.run(hiddenLayerOutput_1)
    b_5 = m5.run(hiddenLayerOutput_1)
    b_6 = m6.run(hiddenLayerOutput_1)

    hiddenLayerOutput_2 = [b_1, b_2, b_3, b_4, b_5, b_6]

    # output_1 = o1.run(hiddenLayerOutput) # Iris-setosa
    output_2 = o2.run(hiddenLayerOutput_2) # Iris-versicolor
    # output_3 = o3.run(hiddenLayerOutput) # Iris-virginica

    actualOutput = [output_2]
    print("desiredOutput: " + str(desiredOutput) + " actualOutput: " + str(actualOutput))
    for i in range(1):
        if actualOutput[i] > 0.5:
            actualOutput[i] = 1
        else:
            actualOutput[i] = 0

    if desiredOutput[1] == actualOutput[0]:
        correctCounter += 1

    # print("desiredOutput: " + str(desiredOutput) + " actualOutput: " + str(actualOutput))

print(str(correctCounter) + " out of " + str(totalTestCounter) + " test cases are correct.")
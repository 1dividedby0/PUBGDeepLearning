""" Project Secondus 'train' API endpoint
    This is the code inside my API that gets called by HTTP Request
"""

import pandas as pd
from random import random
import math
import numpy as np
# AWS S3 Python library
import boto3
import json

# __author__ = Dhruv Mangtani

def sigmoid(x):
    return 1.0/(1.0 + math.exp(-x))

def sigmoid_derivative(output):
    return output * (1.0 - output)

def forwardProp(sample, hidden_layer_weights, output_layer_weights):
    hiddenLayer = []
    # connections to the hidden layer
    for neuron_weights in hidden_layer_weights:
        sum = 0.0
        for input in range(len(neuron_weights)):
            sum += float(sample[input]) * neuron_weights[input]
        hiddenLayer.append(sigmoid(sum))

    output = []
    sum = 0.0
    for output_weight in range(len(output_layer_weights)):
        sum += output_layer_weights[output_weight] * hiddenLayer[output_weight]

    return sigmoid(sum), hiddenLayer

# backpropogate and update the weights using a sample's output vs expected output
def backProp(input_X, output, expected_output, learning_rate, hiddenLayer, hidden_layer_weights, output_layer_weights):
    deltaOutput = (expected_output - output) * sigmoid_derivative(output)

    deltaHidden = []

    for hiddenNeuron in range(len(hiddenLayer)):
        error = output_layer_weights[hiddenNeuron] * deltaOutput
        deltaHidden.append(error * sigmoid_derivative(hiddenLayer[hiddenNeuron]))

    # "i" represents the hidden layer neurons and "j" represents the input layer neurons
    for i in range(len(hidden_layer_weights)):
        for j in range(len(hidden_layer_weights[i])):
            hidden_layer_weights[i][j] += learning_rate * input_X[j] * deltaHidden[i]

    for i in range(len(output_layer_weights)):
        output_layer_weights[i] += learning_rate * hiddenLayer[i] * deltaOutput

    return hidden_layer_weights, output_layer_weights

def train(learning_rate, pubg_data):
    data = pubg_data

    # don't include inputs like 'matchId'
    X = data.iloc[:, 3:28]

    # the actual placement of the player
    expected_Y = data["winPlacePerc"]

    # we don't want string inputs because that is too much of a hassle
    X = X.drop(["matchType"], axis=1)

    # normalize data to 0-1 range
    for input in X.columns:
        max_value = float(X[input].max())
        min_value = float(X[input].min())
        X[input] = (pd.to_numeric(X[input], errors='ignore') - min_value) / (max_value - min_value)

    X = X.values

    # weights from input layer to hidden layer
    hidden_layer_weights = []

    # weights from hidden layer to output layer
    output_layer_weights = []

    # randomize all the weights

    #20 hidden neurons
    for hidden_neuron in range(20):
        hidden_layer_weights.append([])
        for weight in range(3,28):
            # categorical variable
            if weight != 15:
                hidden_layer_weights[hidden_neuron].append(random())

    # weights from hidden layer to output layer
    for weight in range(20):
        output_layer_weights.append(random())

    # train, loop through all samples in data
    for i in range(len(expected_Y)):
        # remove the string input because we don't want to deal with that
        sample = X[i].tolist()

        # forward propogate with the sample
        output, hiddenLayer = forwardProp(sample, hidden_layer_weights, output_layer_weights)

        # update the neural net
        updated_weights = backProp(sample, output, expected_Y[i], learning_rate, hiddenLayer, hidden_layer_weights, output_layer_weights)

        # error
        print(output - expected_Y[i])
        hidden_layer_weights = updated_weights[0]
        output_layer_weights = updated_weights[1]

    return hidden_layer_weights, output_layer_weights

# method that gets called by API
def handler(event, context):

    # download data from AWS s3 and read it using pandas
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket="pubg-train", Key="train.csv")
    data = pd.read_csv(obj["Body"])

    learning_rate = float(event["queryStringParameters"]['learning_rate'])

    hidden_layer_weights, output_layer_weights = train(learning_rate, data)

    output = {"hidden_layer": hidden_layer_weights, "output_layer": output_layer_weights}
    response = {
        "statusCode": 200,
        "headers": {},
        # got this code from:
        #https://stackoverflow.com/questions/46227854/json-stringify-javascript-and-json-dumps-python-not-equivalent-on-a-list
        "body": json.dumps(output, separators=(',',':')),
        "isBase64Encoded": False
    }
    return response


# plt.plot(data["rideDistance"][:100], data["winPlacePerc"][:100], 'ro')
# plt.ylabel("Win Place")
# plt.xlabel("Kill Place")
# plt.show()

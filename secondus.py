""" Project Secondus
    Goal:
    Find out what aspects of a player affect their success the most (find correlations).

    Process:
    Calls an API I made to read csv data, train a neural net, and return its weights.
    Then, uses the weights to determine how important each input variable is and graphs them
    using matplotlib.
"""

import requests
import json
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

# __author__ = Dhruv Mangtani

# train the neural net in the API and return the weights coming from input layer
def getWeights():
    print("Training neural net online...")

    # found this after trial and error
    learning_rate = 0.3

    # HTTP request to 'train' endpoint of my API with learning_rate as an input
    response = requests.get("https://sdnismcaei.execute-api.us-east-1.amazonaws.com/prod/train?learning_rate=" + str(learning_rate))
    jsonResponse = json.loads(response.text)

    # the weights from the trained neural network
    # hiddenLayerWeights is a matrix of the weights from the input layer to the hidden layer
    hiddenLayerWeights = jsonResponse["hidden_layer"]

    return hiddenLayerWeights

# return an array of the averages of the weights coming from each input variable
def averageWeights(hiddenLayerWeights):
    # the importances of each input variable
    importance_inputs = []

    # weightsArray represents a hidden layer neuron
    for weightsArray in hiddenLayerWeights:
        # weight represents a connection/synapse from an input variable to the weightsArray hidden neuron
        for weight in range(len(weightsArray)):
            # if there is no jth input in the importance_inputs array yet then add one
            if weight >= len(importance_inputs):
                importance_inputs.append(math.fabs(weightsArray[weight]))
            else:
                # sum the weights leading from the input variable neuron
                importance_inputs[weight] += math.fabs(weightsArray[weight])

    # Average the sums of weights for all 15 input variables
    for importance in range(len(importance_inputs)):
        importance_inputs[importance] /= 15

    return importance_inputs

# return a color map
def createColorMap():
    # jet style color map from matplotlib
    jet_cmap = plt.get_cmap('jet')

    # allow the color map to normalize values
    cmap_norm = colors.Normalize(vmin=min(importance_inputs), vmax=max(importance_inputs))

    # create a color map
    colorMap = cm.ScalarMappable(norm=cmap_norm, cmap=jet_cmap)

    return colorMap

# plot the importance of each input variable
def plotImportance(input_names, importance_inputs):
    # to be able to convert the importances to colors
    colorMap = createColorMap()

    # horizontal colored bar chart
    bars = plt.barh(input_names, importance_inputs)
    for i in range(len(bars)):
        bars[i].set_color(colorMap.to_rgba(importance_inputs[i]))

    plt.ylabel("Player Feature")
    plt.xlabel("Relative Importance")
    plt.title("How do aspects of a player affect success in PUBG?")

    plt.show()


# weights of neural network from API
hiddenLayerWeights = getWeights()

# names of the player's features/ input variables
input_names = [
                "assists","boosts","damageDealt","DBNOs","headshotKills","heals","killPlace",
                "killPoints","kills","killStreaks","longestKill","matchDuration","maxPlace",
                "numGroups","rankPoints","revives","rideDistance","roadKills","swimDistance",
                "teamKills","vehicleDestroys","walkDistance","weaponsAcquired","winPoints"
                ]

# the values we will plot
importance_inputs = averageWeights(hiddenLayerWeights)

# sort importance_inputs array and input_names array in parallel
# code from https://codereview.stackexchange.com/questions/138702/sorting-two-lists
importance_inputs, input_names = zip(*sorted(zip(importance_inputs, input_names)))

importance_inputs = list(importance_inputs)
input_names = list(input_names)

plotImportance(input_names, importance_inputs)

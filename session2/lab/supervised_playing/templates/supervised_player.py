"""
    This is a PyRat AI file for playing using a trained pytorch classifier.
"""

# PyRat elements
MOVE_DOWN = 'D'
MOVE_LEFT = 'L'
MOVE_RIGHT = 'R'
MOVE_UP = 'U'

# Imports
import numpy as np
import torch
import os
import numpy as np
import pickle
import ast
import sys
import torch.nn as nn

# Load the convert_input function used for generating the dataset
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from generate_dataset import convert_input

# You may need to import additional libraries for your imported models to work
import torch.nn.functional as F

# Global variables
global model

########
# TODO #    Update this constant with the path to the trained model you want to use
########
TRAINED_MODEL_PATH = "path/to/trained_classifier.pkl"

##########################################################################################

"""
    PyRat preprocessing function, executed once at the beginning of the game.
"""

def preprocessing(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, piecesOfCheese, timeAllowed):

    # Here we load the previously trained model
    global model
    contents = pickle.load(open(TRAINED_MODEL_PATH, 'rb'))
    
    # If that's a network, we also need to load weights into the network
    if isinstance(contents, list) :
        global_indent = len(contents[0]) - len(contents[0].lstrip(" "))
        net_code = "\n".join([row[global_indent:] for row in contents[0].split("\n")])
        exec(net_code, globals())
        input_size = convert_input(playerLocation, mazeMap, opponentLocation, mazeHeight, mazeWidth, piecesOfCheese).size
        model = Net(input_size)
        model.load_state_dict(contents[1])
        model.eval()

    # Otherwise we just get the classifier
    else :
        model = contents

##########################################################################################

"""
    PyRat turn function, executed multiple times until the game is over.
    Should return a move.
"""

def turn(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese, timeAllowed):
    
    # Transform the input into the canvas using convert_input 
    input_t = convert_input(playerLocation, mazeMap, opponentLocation, mazeHeight, mazeWidth, piecesOfCheese)
    
    # If the model is a torch classifier
    global model
    if isinstance(model, nn.Module) :
    
        # Predict the next action using the trained model
        input_t = torch.FloatTensor(input_t).reshape(1,-1)
        output = model(input_t)
        action = torch.argmax(output.data, dim=1)[0]

        # Return the action to perform
        return [MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN][action]
    
    # If the model is a sklearn classifier
    else :
    
        # Predict the next action using the trained model
        output = model.predict(input_t.reshape(1,-1))
        action = output[0]

        # Return the action to perform
        return [MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN][action]


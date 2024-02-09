#####################################################################################################################################################
######################################################################## INFO #######################################################################
#####################################################################################################################################################

"""
    This script makes multiple games between two programs, and compares the obtained scores.
    It performs two analyses: a quick average analysis and a formal 1 sample T test.
"""

#####################################################################################################################################################
###################################################################### IMPORTS ######################################################################
#####################################################################################################################################################

# Import PyRat
from pyrat import *

# External imports
import sys
import matplotlib.pyplot as pyplot
import scipy.stats
import os
import numpy
import types
import tqdm

#####################################################################################################################################################
############################################################### VARIABLES & CONSTANTS ###############################################################
#####################################################################################################################################################

"""
    Number of games to make.
"""

NB_GAMES = 500

#####################################################################################################################################################

"""
    Games configuration (default values).
"""

MAZE_WIDTH = 10
MAZE_HEIGHT = 10
CELL_PERCENTAGE = 100
MUD_PERCENTAGE = 0.0
WALL_PERCENTAGE = 0.0
NB_CHEESE = 1
TURN_TIME = 0.0
PREPROCESSING_TIME = 0.0
GAME_MODE = "synchronous"

#####################################################################################################################################################
##################################################################### FUNCTIONS #####################################################################
#####################################################################################################################################################

def run_one_game ( program_1:    types.ModuleType,
                   program_2:    types.ModuleType,
                   seed:         None,
                   nb_cheese:    int=NB_CHEESE,
                   maze_width:   int=MAZE_WIDTH,
                   maze_height:  int=MAZE_HEIGHT,
                   gui:          bool=True
                 ) ->            Tuple[numpy.ndarray, int]:

    """
        This function runs a PyRat game, for a given seed and program, with a given number of cheeses and sizes of the maze, and returns the positions of the cheeses and the winner of the game.
        Parameter gui specifies whether to display the game interface (must be false for large simulations).
        In:
            * program_1:   First program to use in that game.
            * program_2:   Second program to use in that game.
            * seed:        Random seed used to create the game.
            * nb_cheese:   Number of cheeses to catch.
            * maze_width:  Width of the maze.
            * maze_height: Height of the maze.
            * gui:         Whether to display the game interface (True) or not (False).
        Out:
            * cheese_matrix: The positions of the cheeses in the maze.
            * winner:        Winner of the game (1 for program_1, -1 for program_2, and 0 for a draw)
    """
    if gui == True:
        render_mode = "gui"
    else:
        render_mode = "no_rendering"
        
    # Map the functions to the character
    players = [{"name": "Program 1", "team": "1", "preprocessing_function": program_1.preprocessing if "preprocessing" in dir(program_1) else None, "turn_function": program_1.turn, "location": 0},
               {"name": "Program 2", "team": "2", "preprocessing_function": program_2.preprocessing if "preprocessing" in dir(program_2) else None, "turn_function": program_2.turn, "location": maze_width * maze_height - 1}]

    # Customize the game elements
    config = {"maze_width": maze_width,
              "maze_height": maze_height,
              "cell_percentage": CELL_PERCENTAGE,
              "mud_percentage": MUD_PERCENTAGE,
              "wall_percentage": WALL_PERCENTAGE,
              "nb_cheese": nb_cheese,
              "render_mode": render_mode,
              "preprocessing_time": PREPROCESSING_TIME,
              "turn_time": TURN_TIME,
              "game_mode": GAME_MODE,
              "random_seed": seed}
        
    # Set up the game
    game = PyRat(players, **config)
    
    # Find the cheeses
    cheese = game.cheese
    cheese_matrix = numpy.zeros((maze_height*maze_width)) #numpy.zeros((maze_height, maze_width)) # if you want a 2D matrix
    numpy.put(cheese_matrix, cheese, 1)
    
    # Start the game
    stats = game.start()
    
    # Find the winner
    winner = numpy.sign(stats["players"]["Program 1"]["score"] - stats["players"]["Program 2"]["score"])
    
    return cheese_matrix, winner

def run_several_games ( program_1:    types.ModuleType,
                        program_2:    types.ModuleType,
                        nb_cheese:    int=NB_CHEESE,
                        maze_width:   int=MAZE_WIDTH,
                        maze_height:  int=MAZE_HEIGHT,
                        nb_games:     int=NB_GAMES,
                        gui:          bool=False
                      ) ->            Tuple[numpy.ndarray, numpy.ndarray]:
                      
    """
        This function runs a specified number of PyRat games, for a given program, with a given number of cheeses and sizes of the maze, and returns the positions of the cheeses and the winner of the game.
        Parameter gui specifies whether to display the game interface (must be false for large simulations).
        The seeds are set in the enumerate function.
        In:
            * program_1:   First program to use in that game.
            * program_2:   Second program to use in that game.
            * nb_cheese:   Number of cheeses to catch.
            * maze_width:  Width of the maze.
            * maze_height: Height of the maze.
            * gui:         Whether to display the game interface (True) or not (False).
        Out:
            * cheese_matrix: The positions of the cheeses in the maze.
            * winner:        Winner of the game (1 for program_1, -1 for program_2, and 0 for a draw)
    """
    
    X = numpy.zeros((nb_games, maze_width*maze_height)) # Cheeses positions for each game
    Y = numpy.zeros((nb_games)) # Winner of each game
    for idx, seed in enumerate(tqdm.tqdm(range(nb_games), desc="Game", position=0, leave=False)): # Iterating through games
        
        # Run one game, and find the cheese positions and the winner.
        cheese_matrix, winner = run_one_game(program_1, program_2, None, nb_cheese, maze_width, maze_height, gui)
        X[idx] = cheese_matrix
        Y[idx] = winner
        
    return X, Y

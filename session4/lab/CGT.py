#####################################################################################################################################################
######################################################################## INFO #######################################################################
#####################################################################################################################################################

"""
    In this program, we will use combinatorial game theory to build an AI that will play against a greedy opponent.
    The idea is to build an arena of the game, and to induce a playout policy to win.
    We will exploit the knowledge that the opponent is greedy to build a winning strategy.
    This will allow us not to define the entire arena, but only a part of it.
    
    This code has been designed to work properly in a two-player match, in a maze with no mud.
    It is not guaranteed to work in other situations, though it could be adapted.
    
    In this file, all locations where you should write code are indicated by a TODO comment.
    Some functions do not need you to write code, but you should read them carefully to understand what they do.

    Here is your mission:
        1 - Have a quick look at all the functions in this file and in the utils.py file, to understand what they do.
        2 - Find the TODOs in this file and fill them.
            Please read the comments carefully, as they contain important information.
            Run the code from time to time and check that the various functions work as expected.
            You can call the functions from the preprocessing function to test them.
        3 - Once you think everything is ready, run the code and see if you can beat the greedy opponent.
            If it works for one game, try to run the make_2_player_matches.py script to see if it works in general.
            You should get a win or a draw in all games.
            Indeed, if there is no winning strategy, there is still the possibility to play as a greedy player, which guarantees a draw.
        4 - If everything works well, try to change the maze size, the number of pieces of cheese, etc.
            Does the code still work?
            Is there a limit to the size of the maze or the number of pieces of cheese?
"""

#####################################################################################################################################################
###################################################################### IMPORTS ######################################################################
#####################################################################################################################################################

# Import PyRat
from pyrat import *

# Previously developed functions
import sys
import os
lab_commons_path = os.path.join(os.getcwd(), "..", "..")
if lab_commons_path not in sys.path:
    sys.path.append(lab_commons_path)

import lab_commons.AI.greedy as opponent
from lab_commons.utils import bfs, find_route, locations_to_action, get_opponent_name, get_neighbors

#####################################################################################################################################################
##################################################################### FUNCTIONS #####################################################################
#####################################################################################################################################################

def simulate_opponent ( maze:             Union[numpy.ndarray, Dict[int, Dict[int, int]]],
                        maze_width:       int,
                        maze_height:      int,
                        opponent_name:    str,
                        teams:            Dict[str, List[str]],
                        player_locations: Dict[str, int],
                        player_scores:    Dict[str, float],
                        cheese:           List[int],
                        possible_actions: List[str]
                      ) ->                int:

    """
        This function simulates what the opponent would do in the current situation.
        There are many ways to do so, but the simplest is to use the turn function of the opponent.
        Note that this works only if the opponent does not use the memory argument and does nothing in preprocessing that would impact the turns.
        This is the case of the provided greedy opponent.
        Also, we haven't taken mud into consideration, so just use unweighted graphs.
        In:
            * maze:             Map of the maze, as data type described by PyRat's "maze_representation" option.
            * maze_width:       Width of the maze in number of cells.
            * maze_height:      Height of the maze in number of cells.
            * opponent_name:    Name of the opponent.
            * teams:            Recap of the teams of players.
            * player_locations: Locations for all players in the game.
            * player_scores:    Scores for all players in the game.
            * cheese:           List of available pieces of cheese in the maze.
            * possible_actions: List of possible actions.
        Out:
            * opponent_location: Location of the opponent after its turn.
    """

    # TODO
    # Fill this function to simulate the opponent's turn in the current situation
    # I.e., if the opponent was to play, where would it go, given the maze, current players' locations, remaining pieces of cheese, etc.
    # As indicated in the documentation above, there are multiple ways of achieving this
    # You can use the turn function of the opponent directly, or you can write your own code to simulate the opponent's behavior
    # Tip: You can use the function get_neighbors to get the neighboring cells in the maze
    opponent_location = None

    # Return the new location of the opponent
    return opponent_location

#####################################################################################################################################################

def get_updated_cheese_and_scores ( cheese:           List[int],
                                    player_locations: Dict[str, int],
                                    player_scores:    Dict[str, float]
                                  ) ->                Tuple[List[int], Dict[str, float]]:

    """
        This function returns the updated list of cheese and scores after the player and the opponent have played.
        In:
            * cheese:           List of available pieces of cheese in the maze.
            * player_locations: Locations for all players in the game.
            * player_scores:    Scores for all players in the game.
        Out:
            * updated_cheese: List of available pieces of cheese in the maze after the player and the opponent have played.
            * updated_scores: Scores for all players in the game after the player and the opponent have played.
    """

    # Create copies of the cheese and scores
    updated_cheese = cheese.copy()
    updated_scores = player_scores.copy()

    # TODO
    # Fill this function to update the list of cheese and scores after the player and the opponent have played
    # Beware: when updating the cheese, a common mistake is to update the cheese list while iterating over it, as it would modify the indices
    # Therefore, make sure you iterate over the original list of cheese, and update the copy of the list
    # In the rules of PyRat, if a player reaches a cheese alone, they get 1 point
    # If both players reach the same cheese simultaneously, they get 0.5 points each

    # Return the updated cheese and scores
    return updated_cheese, updated_scores

#####################################################################################################################################################

def check_if_over ( cheese:        List[int],
                    player_scores: Dict[str, float]
                  ) ->             Tuple[bool, str]:

    """
        This function checks if the game is over.
        In:
            * cheese:        List of available pieces of cheese in the maze.
            * player_scores: Scores for all players in the game.
        Out:
            * is_over: Boolean indicating if the game is over.
            * winner:  Name of the winner, or None if there is no winner.
    """

    # TODO
    # Fill this function to check if the game is over
    # If the game is over, return True and the name of the winner
    # If there is over and a draw, return True and None
    # If the game is not over, return False and None
    is_over = False
    winner = None

    # Return the result
    return is_over, winner

#####################################################################################################################################################

def simulate_game_up_to_target ( maze:             Union[numpy.ndarray, Dict[int, Dict[int, int]]],
                                 maze_width:       int,
                                 maze_height:      int,
                                 name:             str,
                                 opponent_name:    str,
                                 teams:            Dict[str, List[str]],
                                 player_locations: Dict[str, int],
                                 player_scores:    Dict[str, float],
                                 cheese:           List[int],
                                 possible_actions: List[str],
                                 target:           int
                               ) ->                Tuple[Dict[str, int], Dict[str, float], List[int], bool, str]:

    """
        This function simulates the game up to the point where the player reaches a target.
        In:
            * maze:             Map of the maze, as data type described by PyRat's "maze_representation" option.
            * maze_width:       Width of the maze in number of cells.
            * maze_height:      Height of the maze in number of cells.
            * name:             Name of the player controlled by this function.
            * opponent_name:    Name of the opponent.
            * teams:            Recap of the teams of players.
            * player_locations: Locations for all players in the game.
            * player_scores:    Scores for all players in the game.
            * cheese:           List of available pieces of cheese in the maze.
            * possible_actions: List of possible actions.
            * target:           Target to reach.
        Out:
            * new_player_locations: Locations for all players in the game after the player and the opponent have played.
            * new_player_scores:    Scores for all players in the game after the player and the opponent have played.
            * new_cheese:           List of available pieces of cheese in the maze after the player and the opponent have played.
            * is_over:              Boolean indicating if the game is over.
            * winner:               Name of the winner, or None if there is no winner.
    """

    # Find the shortest path for the player to reach the target
    # We use the BFS provided in the greedy opponent file
    _, routing_table = bfs(player_locations[name], maze, target)
    route = find_route(routing_table, player_locations[name], target)
    del route[0]

    # Update the configuration of the game until the player reaches the target
    new_player_locations = player_locations.copy()
    new_player_scores = player_scores.copy()
    new_cheese = cheese.copy()
    while len(route) > 0:

        # Make one move in the right direction
        new_player_location = route.pop(0)
        new_opponent_location = simulate_opponent(maze, maze_width, maze_height, opponent_name, teams, new_player_locations, new_player_scores, new_cheese, possible_actions)
        new_player_locations = {name: new_player_location, opponent_name: new_opponent_location}

        # Check state of the game and abort if game is over
        new_cheese, new_player_scores = get_updated_cheese_and_scores(new_cheese, new_player_locations, new_player_scores)
        is_over, winner = check_if_over(new_cheese, new_player_scores)
        if is_over:
            break

    # Return the new configuration of the game
    return new_player_locations, new_player_scores, new_cheese, is_over, winner

#####################################################################################################################################################
##################################################### EXECUTED ONCE AT THE BEGINNING OF THE GAME ####################################################
#####################################################################################################################################################

def preprocessing ( maze:             Union[numpy.ndarray, Dict[int, Dict[int, int]]],
                    maze_width:       int,
                    maze_height:      int,
                    name:             str,
                    teams:            Dict[str, List[str]],
                    player_locations: Dict[str, int],
                    cheese:           List[int],
                    possible_actions: List[str],
                    memory:           threading.local
                  ) ->                None:

    """
        This function is called once at the beginning of the game.
        It is typically given more time than the turn function, to perform complex computations.
        Store the results of these computations in the provided memory to reuse them later during turns.
        To do so, you can crete entries in the memory dictionary as memory.my_key = my_value.
        In:
            * maze:             Map of the maze, as data type described by PyRat's "maze_representation" option.
            * maze_width:       Width of the maze in number of cells.
            * maze_height:      Height of the maze in number of cells.
            * name:             Name of the player controlled by this function.
            * teams:            Recap of the teams of players.
            * player_locations: Locations for all players in the game.
            * cheese:           List of available pieces of cheese in the maze.
            * possible_actions: List of possible actions.
            * memory:           Local memory to share information between preprocessing, turn and postprocessing.
        Out:
            * None.
    """

    # Find the opponent's name
    opponent_name = get_opponent_name(teams, name)

    # Initial state of the game
    player_scores = {player: 0 for player in player_locations}
    state = (player_locations.copy(), cheese.copy(), player_scores.copy())

    # Configurations will be put in a list
    parent_state = None
    target = player_locations[name]
    is_over = False
    winner = None
    configurations = [(state, parent_state, target, is_over, winner)]

    # Explore all possible states
    # The following code is basically just a traversal on the graph of game configurations
    final_configuration = None
    states_routing_table = {}
    branches = 0
    while len(configurations) > 0:

        # Pop a state and ignore duplicates that may have been added
        state, parent_state, target, is_over, winner = configurations.pop(0)
        state_player_locations, state_cheese, state_player_scores = state
        if str(state) not in states_routing_table:

            # Update the routing table
            # Here, we use string representations of the state as keys, to avoid issues with the fact that dictionaries cannot use lists as keys
            states_routing_table[str(state)] = (parent_state, target)
        
            # If the game is over, we remember the state if a draw (current best final state found) or a win (and we stop here the search)
            if is_over:
                branches += 1
                if winner != opponent_name:
                    final_configuration = (state, winner)
                if winner == name:
                    break
                continue

            # TODO
            # If the game is not over, we explore all possible next targets
            # In other words, each choice of a piece of cheese to go to leads to a new configuration of the game
            # We add all these new configurations to the list of configurations to explore, just like if these were neighbors of the current configuration in a traversal
            # Note that we do not need to explore the entire graph of game configurations, but only a part of it
            # Indeed, the stopping condition above ensures that we stop the search as soon as we find a winning strategy
            # This is only possible because the opponent is greedy, and thus we can exploit this knowledge to build a winning strategy
            # In a more general case, we would need to explore the entire graph of game configurations, which would be much more costly

    # Find the route to the final state
    memory.route = []
    final_state, winner = final_configuration
    while final_state != None:
        final_state, target = states_routing_table[str(final_state)]
        memory.route.insert(0, target)

    # Print a summary
    print("Found the best route in", branches, "branches")
    print("Best route found is", memory.route)
    print("Expected outcome is a", "win" if winner == name else "draw" if winner == None else "loss")

#####################################################################################################################################################
######################################################### EXECUTED AT EACH TURN OF THE GAME #########################################################
#####################################################################################################################################################

def turn ( maze:             Union[numpy.ndarray, Dict[int, Dict[int, int]]],
           maze_width:       int,
           maze_height:      int,
           name:             str,
           teams:            Dict[str, List[str]],
           player_locations: Dict[str, int],
           player_scores:    Dict[str, float],
           player_muds:      Dict[str, Dict[str, Union[None, int]]],
           cheese:           List[int],
           possible_actions: List[str],
           memory:           threading.local
         ) ->                str:

    """
        This function is called at every turn of the game and should return an action within the set of possible actions.
        You can access the memory you stored during the preprocessing function by doing memory.my_key.
        You can also update the existing memory with new information, or create new entries as memory.my_key = my_value.
        In:
            * maze:             Map of the maze, as data type described by PyRat's "maze_representation" option.
            * maze_width:       Width of the maze in number of cells.
            * maze_height:      Height of the maze in number of cells.
            * name:             Name of the player controlled by this function.
            * teams:            Recap of the teams of players.
            * player_locations: Locations for all players in the game.
            * player_scores:    Scores for all players in the game.
            * player_muds:      Indicates which player is currently crossing mud.
            * cheese:           List of available pieces of cheese in the maze.
            * possible_actions: List of possible actions.
            * memory:           Local memory to share information between preprocessing, turn and postprocessing.
        Out:
            * action: One of the possible actions, as given in possible_actions.
    """

    # Define the next target
    if memory.route[0] == player_locations[name]:
        del memory.route[0]

    # Move toward it
    _, routing_table = bfs(player_locations[name], maze, memory.route[0])
    route_to_target = find_route(routing_table, player_locations[name], memory.route[0])
    action = locations_to_action(player_locations[name], route_to_target[1], maze_width)
    return action

#####################################################################################################################################################
######################################################################## GO! ########################################################################
#####################################################################################################################################################

if __name__ == "__main__":

    # Map the functions to the character
    players = [{"name": "CGT",
                    "team": "You",
                    "skin": "rat",
                    "preprocessing_function": preprocessing,
                    "turn_function": turn},
               {"name": "Greedy",
                    "team": "Opponent",
                    "skin": "python",
                    "preprocessing_function": opponent.preprocessing if "preprocessing" in dir(opponent) else None,
                    "turn_function": opponent.turn,
                    "postprocessing_function": opponent.postprocessing if "postprocessing" in dir(opponent) else None}]

    # Customize the game elements
    config = {"maze_width": 15,
              "maze_height": 15,
              "mud_percentage": 0.0,
              "nb_cheese": 7,
              "game_mode": "synchronous"}

    # Start the game
    game = PyRat(players, **config)
    stats = game.start()

    # Show statistics
    print(stats)

#####################################################################################################################################################
#####################################################################################################################################################
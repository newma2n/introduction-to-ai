#################################################################################################################
###################################################### INFO #####################################################
#################################################################################################################

"""
    This program performs a depth-first search in the maze, starting from the player location.
    The search is performed in the "preprocessing" function at the beginning of the game.
    Actions are then returned one by one in the "turn" function.
    Note that the search is made by the algorithm, like in the head of the player, before the game starts.
    For this reason, your character should only move along the returned path.
"""

#################################################################################################################
#################################################### IMPORTS ####################################################
#################################################################################################################

import heapq

# Import PyRat
from pyrat import *

# Previously developed functions
from AI.utils.BFS import traversal, find_route, locations_to_actions

#################################################################################################################
################################################### FUNCTIONS ###################################################
#################################################################################################################

def dijkstra ( source: int,
          graph:  Union[numpy.ndarray, Dict[int, Dict[int, int]]]
        ) ->      Tuple[Dict[int, int], Dict[int, Union[None, int]]]:

    """
        A DFS is a particular traversal where vertices are explored in the inverse order where they are added to the structure.
        In:
            * source: Vertex from which to start the traversal.
            * graph:  Graph on which to perform the traversal.
        Out:
            * distances_to_explored_vertices: Dictionary where keys are explored vertices and associated values are the lengths of the paths to reach them.
            * routing_table:                  Routing table to allow reconstructing the paths obtained by the traversal.
    """
    
    # Function to create an empty LIFO, encoded as a list
    def _create_structure ():
        return []

    # Function to add an element to the LIFO (elements enter by the end)
    def _push_to_structure (structure, element):
        heapq.heappush(structure, element)

    # Function to extract an element from the LIFO (elements exit by the end)
    def _pop_from_structure (structure):
        return heapq.heappop(structure)
    
    # Perform the traversal
    distances_to_explored_vertices, routing_table = traversal(source, graph, _create_structure, _push_to_structure, _pop_from_structure)
    return distances_to_explored_vertices, routing_table

#################################################################################################################
################################### EXECUTED ONCE AT THE BEGINNING OF THE GAME ##################################
#################################################################################################################

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

    # We perform a traversal from the initial location and save the sequence of actions
    _, routing_table = dijkstra(player_locations[name], maze)
    route = find_route(routing_table, player_locations[name], cheese[0])
    memory.actions_to_perform = locations_to_actions(route, maze_width)

#################################################################################################################
####################################### EXECUTED AT EACH TURN OF THE GAME #######################################
#################################################################################################################

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

    # Apply actions in order
    action = memory.actions_to_perform.pop(0)
    return action

#################################################################################################################
###################################################### GO ! #####################################################
#################################################################################################################

if __name__ == "__main__":

    # Map the functions to the character
    players = [{"name": "Dijkstra", "preprocessing_function": preprocessing, "turn_function": turn}]

    # Customize the game elements
    config = {"maze_width": 15,
              "maze_height": 11,
              "mud_percentage": 40.0,
              "nb_cheese": 1,
              "trace_length": 1000}

    # Start the game
    game = PyRat(players, **config)
    stats = game.start()

    # Show statistics
    print(stats)

#################################################################################################################
#################################################################################################################
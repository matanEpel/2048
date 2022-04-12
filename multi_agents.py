import numpy as np
import abc
import util
from game import Agent, Action
from game_state import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"
        board = game_state.board
        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.

        """

        # Useful information you can extract from a GameState (game_state.py)

        successor_game_state = current_game_state.generate_successor(action=action)
        board = successor_game_state.board
        max_tile = successor_game_state.max_tile
        score = successor_game_state.score

        """
        DESCRIPTION: The idea behind this evaluation function is, as you suggested - combining some heuristics
        together in order to create a better one. All the heuristics are normalized to 1 in order to give them the same
        effect (and then we can multiply each one by a constant in order to give each one a different weight.
        The three heuristics are:
        1. "symmetric formation", as an individual who played tons of 2048 games in the past, the best strategy
        is to create a symmetric formation. Meaning - top left is the highest tile, and as we go away from it
        the values decrease. For example:
        64 32 16 0
        32 16 0  0
        16 0  0  0
    
        2. "free tiles" we want to have a lot of "free tile", so we give score for the amount of
        free tiles divided by the amount of possible free tiles.
    
        3. highest to the corner. The symmetric formation is nice, but we want to keep the highest
        tile in one of the corners even if when it is not the formation is less symmetric. So, we gave
        "bonus" points for boards where the highest is in the corner.
        
        The average highest tile over 1000 games using Reflex Agent with this function is 267.
        """

        board = successor_game_state.board

        # the symmetric formation we have:
        scores1 = np.array([[6, 5, 4, 3], [5, 4, 3, 2], [4, 3, 2, 1], [3, 2, 1, 0]])
        scores2 = np.array([[3, 4, 5, 6], [2, 3, 4, 5], [1, 2, 3, 4], [0, 1, 2, 3]])
        scores3 = np.array([[3, 2, 1, 0], [4, 3, 2, 1], [5, 4, 3, 2], [6, 5, 4, 3]])
        scores4 = np.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])

        scores = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6]

        # calculating the max symmetric score:
        snake = 0
        for s in [scores1, scores2, scores3, scores4]:
            snake = max(snake, np.sum(np.multiply(s, board)))

        # normalizing:
        max_possible = 0
        data = np.sort(board.flatten())
        for i in range(0, len(data)):
            max_possible += data[i] * scores[i]
        snake_score = snake / max_possible

        # calculating the percentage of free tiles:
        free_tiles_factor = 15 - np.log2(successor_game_state.max_tile)
        free_tiles_score = (16 - np.count_nonzero(board)) / free_tiles_factor

        # adding the "max tile to the corner" bonus
        max_bonus = 0
        if max([board[0, 0], board[0, 3], board[3, 0], board[3, 3]]) == successor_game_state.max_tile:
            max_bonus = 1
        return snake_score + free_tiles_score + max_bonus * 2


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):
    def get_action(self, game_state):
        """
        get action returns the next desired action using the minmax algorithm.
        :param game_state: the current game state
        :return: the action we want to do
        """

        legal_moves = game_state.get_agent_legal_actions()
        # Choose one of the best actions
        scores_max = [self.get_action_minimax(game_state.generate_successor(0, action), 0, False) for action in legal_moves]
        best_score = max(scores_max)
        best_indices = [index for index in range(len(scores_max)) if scores_max[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        return legal_moves[chosen_index]

    def get_action_minimax(self, game_state: GameState, curDepth, maxTurn):
        """
        this function returns the minamx score of a state (when going curDepth down)
        :param game_state: the curr game state
        :param curDepth: the curr depth
        :param maxTurn: is it max's turn or min's turn
        :return: the action cost
        """

        # there are no more states - we lose:
        if maxTurn and len(game_state.get_agent_legal_actions()) == 0:
            return -np.inf
        elif not maxTurn and len(game_state.get_opponent_legal_actions()) == 0:
            return -np.inf

        # if we are at the max depth and we are in the opponent turn - we are finishing:
        if curDepth == self.depth - 1 and not maxTurn:
            legal_moves = game_state.get_opponent_legal_actions()
            # calculating score of each state and getting the min:
            scores_max = [self.evaluation_function(game_state.generate_successor(1, action)) for action in legal_moves]
            return min(scores_max)

        # if it is max's turn:
        if maxTurn:
            # getting all options and getting the max:
            legal_moves = game_state.get_agent_legal_actions()
            scores_max = [self.get_action_minimax(game_state.generate_successor(0, action), curDepth, False) for action in
                      legal_moves]
            return max(scores_max)
        else:
            # if it is the opponent turn, we choose the min of all the options:
            legal_moves = game_state.get_opponent_legal_actions()
            scores_max = [self.get_action_minimax(game_state.generate_successor(1, action), curDepth + 1, True) for action
                      in legal_moves]
            return min(scores_max)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        get action returns the next desired action using the minmax algorithm.
        :param game_state: the current game state
        :return: the action we want to do
        """
        legal_moves = game_state.get_agent_legal_actions()
        # Choose one of the best actions
        scores_alpha = [self.get_action_alpha_beta(game_state.generate_successor(0, action), 0, False, float("-inf"), float("inf"))
                    for action in legal_moves]
        best_score = max(scores_alpha)
        best_indices = [index for index in range(len(scores_alpha)) if scores_alpha[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        return legal_moves[chosen_index]

    def get_action_alpha_beta(self, game_state: GameState, curDepth, maxTurn, alpha, beta):
        """
        returns the best score of the current state using alpha-beta minmax
        :param game_state: current game state
        :param curDepth: the current depth
        :param maxTurn: is it max's turn?
        :param alpha: alpha value
        :param beta: beta value
        :return: the score
        """
        # there are no more states - we lose:
        if maxTurn and len(game_state.get_agent_legal_actions()) == 0:
            return -np.inf
        elif not maxTurn and len(game_state.get_opponent_legal_actions()) == 0:
            return -np.inf

        # if we are at the max depth and we are in the opponent turn - we are finishing:
        if curDepth == self.depth - 1 and not maxTurn:
            legal_moves = game_state.get_opponent_legal_actions()
            scores_alpha = [self.evaluation_function(game_state.generate_successor(1, action)) for action in legal_moves]
            return min(scores_alpha)

        # if it is max's turn:
        if maxTurn:
            legal_moves = game_state.get_agent_legal_actions()
            max_val = np.float('-inf')
            # running through all the moves and breaking if the alpha-beta condition is satisfied
            for action in legal_moves:
                curr_val = self.get_action_alpha_beta(game_state.generate_successor(0, action), curDepth, False, alpha,
                                                      beta)
                max_val = max(curr_val, max_val)
                alpha = max(alpha, max_val)
                if beta <= alpha:
                    break
            return max_val

        else:
            legal_moves = game_state.get_opponent_legal_actions()
            min_val = np.float('inf')
            # running through all the moves and breaking if the alpha-beta condition is satisfied
            for action in legal_moves:
                curr_val = self.get_action_alpha_beta(game_state.generate_successor(1, action), curDepth + 1, True,
                                                      alpha,
                                                      beta)
                min_val = min(curr_val, min_val)
                beta = min(beta, min_val)
                if beta <= alpha:
                    break

            return min_val


def weight(action):
    """
    returns the chance of getting this tile on the board
    :param action: the action
    :return: 0.9 (90%) if it is 2, 0.1 (10%) if it is false
    """
    if action.value == 2:
        return 0.9
    else:
        return 0.1


class ExpectimaxAgent(MultiAgentSearchAgent):
    def get_action(self, game_state):
        """
        get action returns the next desired action using the expectimax algorithm.
        :param game_state: the current game state
        :return: the action we want to do
        """
        legal_moves = game_state.get_agent_legal_actions()
        # Choose one of the best actions
        scores = [self.get_action_expectimax(game_state.generate_successor(0, action), 0, False) for action in
                  legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        return legal_moves[chosen_index]

    def get_action_expectimax(self, game_state: GameState, curDepth, maxTurn):
        # there are no more states - we lose:
        if maxTurn and len(game_state.get_agent_legal_actions()) == 0:
            return -np.inf
        elif not maxTurn and len(game_state.get_opponent_legal_actions()) == 0:
            return -np.inf

        # if we are at the max depth and we are in the opponent turn - we are finishing:
        if curDepth == self.depth - 1 and not maxTurn:
            legal_moves = game_state.get_opponent_legal_actions()
            scores = [self.evaluation_function(game_state.generate_successor(1, action))*weight(action)
                      for action in legal_moves]
            # the *2 is because we already did the averaging for the 2,4 options in the same tile:
            return np.average(scores)*2

        # max's turn:
        if maxTurn:
            legal_moves = game_state.get_agent_legal_actions()
            scores_expect = [self.get_action_expectimax(game_state.generate_successor(0, action), curDepth, False) for
                             action in legal_moves]
            # returning the best possible move:
            return max(scores_expect)

        else:
            # min's turn
            legal_moves = game_state.get_opponent_legal_actions()
            scores_expect = [self.get_action_expectimax(game_state.generate_successor(1, action), curDepth + 1, True)*
                             weight(action) for action in legal_moves]
            # returning the expected score of the current state:
            return np.average(scores_expect)


def smoothness(board):
    """
    measures the smoothness of the boar
    Meaning, how much neighbor tiles are similar
    :param board: the board
    :return: the smoothness score
    """
    score = 0
    # applying log in order to the distance (in amount of tiles needed to be joined) from 0
    board_log = np.log2(board)
    board_log[board_log == -np.inf] = 0

    # for each location of the board - adding the distance from it's neighbors to the score
    for i in range(1, 4):
        for j in range(1, 4):
            if board_log[i, j] != 0:
                if board_log[i, j - 1] != 0:
                    score -= np.abs(board_log[i, j] - board_log[i, j - 1])
                if board_log[i - 1, j] != 0:
                    score -= np.abs(board_log[i, j] - board_log[i - 1, j])
    # adding the top row and left column:
    for i in range(1, 4):
        if board_log[0, i] != 0 and board_log[0, i - 1] != 0:
            score -= np.abs(board_log[0, i] - board_log[0, i - 1])
        if board_log[i, 0] != 0 and board_log[i - 1, 0] != 0:
            score -= np.abs(board_log[i, 0] - board_log[i - 1, 0])
    return score


def monotonicity(board):
    """
    measures the monotonicity of the board
    meaning - how much the up/down and the left/right directions are monotonic
    :param board: the current board
    :return: the monotonicity score
    """
    # applying log in order to the distance (in amount of tiles needed to be joined) from 0
    board_log = np.log2(board)

    up = 0
    down = 0
    left = 0
    right = 0

    for i in range(4):
        row = board_log[i, :]
        row = row[row != -np.inf]
        collum = board_log[:, i]
        collum = collum[collum != -np.inf]

        # adding the monotonicity between two tiles for the right direction (up/down, left/right):
        for v in np.diff(row):
            if v > 0:
                left -= v
            else:
                right += v
        for v in np.diff(collum):
            if v > 0:
                up -= v
            else:
                down += v
    # print(up, down, left, right)
    return max([up, down]) + max([left, right])


def better_evaluation_function(current_game_state):
    """
    the evaluation function. description for the logic behind it is in the readme
    :param current_game_state: the current game state
    :return: the evaluation value
    """
    board = current_game_state.board

    # if we do not have more moves: we don't want this option!!!
    if len(current_game_state.get_agent_legal_actions) == 0:
        return -np.inf

    smoothness_score = smoothness(board)
    monotonicity_score = monotonicity(board)

    # adding score for the amount of empty cells:
    empty_score = np.log2(16 - np.count_nonzero(board))

    # adding/subtracting score for the max tile to be in one of the corners
    if current_game_state.max_tile in [board[0, 0], board[0, 3], board[3, 0], board[3, 3]]:
        max_score = np.log2(current_game_state.max_tile)
    else:
        max_score = -np.log2(current_game_state.max_tile)

    # combining all the measurements:
    return 0.1 * smoothness_score + monotonicity_score + 1.8 * empty_score + max_score


# Abbreviation
better = better_evaluation_function

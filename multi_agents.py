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
        Well, when we started the exercise, we've implemented a basic function
        that takes into account these three parameters:
        1. whether the max tile of the successor state is bigger than the current state
        2. whether the amount of free locations is bigger.
        3. whether the score is higher.
        and giving "bonus" points to the evaluation accordingly.
        However, this approach gave us an average of 187 (max tile),
        And a distribution of [32: 0.2%, 64: 10%, 128: 45%, 256: 42%, 512: 33%]
        which is not very good.
        So, we "commented this function (below).
        
        Then, we created the "better_evaluation_function" for the minimax algorithm,
        and after a lot of thinking it is a much better function. So, we've decided to
        use it instead.
        The stats for this function when running the reflex agents are (after running on 1000 games):
        Average: 268 (max tile)
        Distribution: [32: 0.2%, 64: 2.9%, 128: 17.8%, 256: 64.4%, 512: 15%, 1024: 0.3%]
        Which is much better than the previous one
        """
        """
        max_bonus = 0
        if current_game_state.max_tile < max_tile:
            max_bonus = 1

        free_bonus = np.count_nonzero(current_game_state.board) - np.count_nonzero(board) - 1

        score_bonus = 0
        if score > current_game_state.score:
            score_bonus = 1

        return score_bonus + free_bonus + max_bonus
        """

        return better_evaluation_function(successor_game_state)


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
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        """*** YOUR CODE HERE ***"""

        legal_moves = game_state.get_agent_legal_actions()
        # Choose one of the best actions
        print(legal_moves)
        scores = [self.get_action_minimax(game_state.generate_successor(0, action), 0, True) for action in legal_moves]
        print(scores)
        best_score = self.get_action_minimax(game_state, 0, True)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        return legal_moves[chosen_index]

    def get_action_minimax(self, game_state: GameState, curDepth, maxTurn):
        print("entered mini max")
        if curDepth == self.depth - 1:
            legal_moves = game_state.get_agent_legal_actions()
            scores = [self.evaluation_function(game_state, action) for action in legal_moves]
            return max(scores)

        if maxTurn:
            legal_moves = game_state.get_agent_legal_actions()
            scores = [self.get_action_minimax(game_state.generate_successor(0, action), curDepth+1, False) for action in legal_moves]
            return max(scores)

        else:
            return self.get_action_minimax(game_state.generate_successor(1), curDepth, True)





class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        """*** YOUR CODE HERE ***"""
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        """*** YOUR CODE HERE ***"""
        util.raiseNotDefined()


def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

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

    """
    "*** YOUR CODE HERE ***"
    board = current_game_state.board

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
    free_tiles_factor = 15 - np.log2(current_game_state.max_tile)
    free_tiles_score = (16 - np.count_nonzero(board)) / free_tiles_factor

    # adding the "max tile to the corner" bonus
    max_bonus = 0
    if max([board[0,0], board[0,3], board[3,0], board[3,3]]) == current_game_state.max_tile:
        max_bonus = 1
    return snake_score + free_tiles_score + max_bonus*2


# Abbreviation
better = better_evaluation_function

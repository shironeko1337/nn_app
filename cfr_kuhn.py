import util
import numpy as np
from collections import defaultdict
# http://modelai.gettysburg.edu/2013/cfr/cfr.pdf
# CFR for kuhn poker, recursive monte carlo training

ACTIONS = 'pb'  # pass, bet
ACTIONS_N = 2
node_map = defaultdict(lambda: Node())

# a node in the game state tree which generates a new branch whenever
# any player plays an action under any new info set
class Node:
    def __init__(self):
        self.regret_sum = np.zeros(ACTIONS_N)
        self.strategy = np.zeros(ACTIONS_N)
        self.strategy_sum = np.zeros(ACTIONS_N)

    # 1. get the current information set mixed strategy through regret-matching
    # similar to RPS
    # 2. adding to the strategy sum of the current node
    def getStrategy(self, realization_weight):
        result = util.rectified_normalize(self.regret_sum, 1.0 / ACTIONS_N)
        self.strategy_sum += result * realization_weight
        return result

    # similar to RPS
    def getAverageStrategy(self):
        return util.normalize(self.strategy_sum, 1.0 / ACTIONS_N)

    def __str__(self):
        return self.info + str(self.getAverageStrategy())

# from history, return the results in a tuple
# - if it's terminal
# - player i's utility
def getTerminalResult(initial_state, history, i):
    if len(history) > 1:
        player_card, opponent_card = initial_state[i], initial_state[i ^ 1]
        last = history[-1]
        last_two = history[-2:]
        if last == 'p':
            utility = [1, -1][player_card >
                              opponent_card] if last_two == 'pp' else 1
            return 1, utility
        elif last_two == 'bb':
            utility = [2, -2][player_card > opponent_card]
            return 1, utility
        else:
            return 0, 0
    return 0, 0

# initial_state cards array, initial_state[0] is dealt to player0, initial_state[1] to player1
# history, initially have zero length, we can get the current turn index and know which player
#   is playing at this round, game always starts with player 0. history is public
#   so we can simply use the player's initial card + current history to get
#   the info of the player. Normally initial card dealing should be counted into
#   history but here we separate it for simplicity.
# p0 conditional probability for Player0 to reach the h with his own contribution
# p1 similar, p0 p1 together to make sure p0 * p1 = p(h)
# p0 and p1 all start with 1
def cfr(initial_state, history, p0, p1):
    turn_index = len(history)
    player = turn_index & 1
    history_probability = [p0, p1][player]
    isTerminal, playerUtility = getTerminalResult(
        initial_state, history, player)

    if isTerminal:
        return playerUtility

    # we don't store the player index since size of history has it
    info = str(initial_state[player]) + history
    node = node_map[info]
    node.info = info
    strategy = node.getStrategy(history_probability)
    utility = [0, 0]

    # sum up the counterfactual regret for each action
    nodeUtility = 0
    for action, action_str in enumerate(ACTIONS):
        if player == 0:
            utility[action] = -cfr(initial_state,
                                   history + action_str,
                                   p0 * strategy[action],
                                   p1)
        else:
            utility[action] = -cfr(initial_state,
                                   history + action_str,
                                   p0,
                                   p1 * strategy[action])
        nodeUtility += strategy[action] * utility[action]

    # here nodeUtility is the weighted "best" utility, and we add up
    # the regret for each action to regret sum
    for action, action_str in enumerate(ACTIONS):
        node.regret_sum[action] += history_probability * \
            (utility[action] - nodeUtility)

    return nodeUtility

def train(iterations):
    cards = np.array([0, 1, 2])
    for _ in range(iterations):
        np.random.shuffle(cards)
        cfr(cards, "", 1, 1)

    for info, average_strategy in node_map.items():
        print(average_strategy)


train(100)

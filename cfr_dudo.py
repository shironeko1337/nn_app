# http://modelai.gettysburg.edu/2013/cfr/cfr.pdf
# there is a flaw in the formula of s(n,r) should be 5n - [n/2] + r - 7
# https://en.wikipedia.org/wiki/Dudo
import util
import numpy as np
from sortedcontainers import SortedDict

# action of 0 - 11: claim
# action of 12: dudo
ACTIONS_N = 13
ACTIONS = list(range(ACTIONS_N))
MAX_STRENGTH = 11
node_map = SortedDict()

def encode_action(strength):
    return hex(strength)[2:]

def decode_action(action):
    action = int(action, 16)
    dudo = action == 12
    count = action // 6 + 1
    rank = (action % 6 + 1) % 6
    return dudo, count, rank

# game ends when the last action is dudo
# 1 if challenged player is not bluffing
def get_terminal_result(initial_state, history):
    if len(history) > 1:
        dudo, count, rank = decode_action(history[-1])
        if dudo:
            _, prev_count, prev_rank = decode_action(history[-2])
            return 1, [-1, 1][initial_state['total'][prev_rank] >= prev_count]
    return 0, 0

def get_next_state(history, action, strategy, p0, p1):
    player = len(history) & 1
    next_history = history + encode_action(action)
    next_p0 = p0 if player else p0 * strategy[action]
    next_p1 = p1 * strategy[action] if player else p1
    return next_history, next_p0, next_p1

class Node:
    def __init__(self, info):
        self.info = info
        self.regret_sum = np.zeros(ACTIONS_N)
        self.strategy_sum = np.zeros(ACTIONS_N)

    def get_strategy(self, weight):
        res = util.rectified_normalize(self.regret_sum, 1.0 / ACTIONS_N)
        self.strategy_sum += res * weight
        return res

    def get_average_strategy(self):
        return util.normalize(self.strategy_sum, 1.0 / ACTIONS_N)

def cfr(initial_state, history, p0, p1):
    player, opponent = len(history) & 1, len(history) & 1 ^ 1
    cum_probability, cum_others_probability = [
        p0, p1][player], [p0, p1][opponent]
    is_terminal, utility = get_terminal_result(initial_state, history)
    if is_terminal:
        return utility

    # get the info set known to the current player and the node in game tree
    info = initial_state['p1' if player else 'p0'] + history

    if info in node_map:
        node = node_map[info]
    else:
        node = node_map[info] = Node(info)

    # get available actions
    minimal_next_action_strength = int(
        history[-1], 16) + 1 if len(history) else 0

    available_actions = list(
        range(minimal_next_action_strength, MAX_STRENGTH + 1))
    if len(history):
        available_actions += [12]

    # get node utility and action utility
    node_utility = 0
    strategy = node.get_strategy(cum_probability)
    utility = [0] * ACTIONS_N
    for action in available_actions:
        next_history, next_p0, next_p1 = get_next_state(
            history, action, strategy, p0, p1)
        utility[action] = -cfr(initial_state, next_history, next_p0, next_p1)
        node_utility += utility[action] * strategy[action]

    # update regret sum
    for action in available_actions:
        node.regret_sum[action] += cum_others_probability * \
            (utility[action] - node_utility)
    return sum(utility)


def train(iterations):
    for _ in range(iterations):
        p0, p1 = np.random.randint(6), np.random.randint(6)
        total = [0] * 6
        total[p0] += 1
        total[p1] += 1
        initial_state = {
            'p0': str(p0),
            'p1': str(p1),
            'total': total
        }

        cfr(initial_state, "", 1, 1)
    print(len(node_map.items()))
train(10)

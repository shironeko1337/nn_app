# http://modelai.gettysburg.edu/2013/cfr/cfr.pdf
# CFR for RPS game

import numpy as np

ACTIONS = [0,1,2] # 0 = Rock, 1 = Paper, 2 = Scissor
ACTIONS_N = len(ACTIONS)

# strategy should be normalized, and be a randomized strategy when regret sum is negative
def getStartegy(regrets):
    regrets = np.where(regrets > 0, regrets, 0)
    return getAverageStrategy(regrets)

def getAverageStrategy(regrets):
    regretsSum = np.sum(regrets)
    return regrets / regretsSum if regretsSum > 0 else np.array([1.0 / ACTIONS_N]*ACTIONS_N)

# get the utility of one's action by the set of all actions
def getUtility(action,op_action):
    return 0 if action == op_action else [1,-1][((op_action - 1) % ACTIONS_N) == action]

def getAction(strategy):
    return np.random.choice(list(range(len(strategy))),size = 1, p=strategy)[0]

# train an agent given a constant strategy of opponent
def train_one(iteration):
    strategy_op = np.array([.4,.3,.3])
    regrets = np.zeros(ACTIONS_N)
    strategy_sum = np.zeros(ACTIONS_N)

    for _ in range(iteration):
        # calculate average strategy by regrets
        strategy = getStartegy(regrets)
        strategy_sum += strategy
        # get action by strategy
        action = getAction(strategy)
        op_action = getAction(strategy_op)
        # get utility by action
        utilities = np.array([getUtility(a,op_action) for a in ACTIONS])
        # accumulate regrets for each action
        regrets = np.array([regrets[i] + utilities[a] - utilities[action] for i,a in enumerate(ACTIONS)])

    return getAverageStrategy(strategy_sum)

def train_both(iteration):
    strategy_op = np.array([.4,.3,.3])
    regrets = np.zeros(ACTIONS_N)
    regrets_op = np.zeros(ACTIONS_N)
    strategy_sum = np.zeros(ACTIONS_N)
    strategy_sum_op = np.zeros(ACTIONS_N)

    for _ in range(iteration):
        # calculate average strategy by regrets
        strategy = getStartegy(regrets)
        strategy_op = getStartegy(regrets_op)
        strategy_sum += strategy
        strategy_sum_op += strategy_op
        # get action by strategy
        action = getAction(strategy)
        action_op = getAction(strategy_op)
        # get utility by action
        utilities = np.array([getUtility(a,action_op) for a in ACTIONS])
        utilities_op = np.array([getUtility(a,action) for a in ACTIONS])
        # accumulate regrets for each action
        regrets = np.array([regrets[i] + utilities[a] - utilities[action] for i,a in enumerate(ACTIONS)])
        regrets_op = np.array([regrets_op[i] + utilities_op[a] - utilities_op[action_op] for i,a in enumerate(ACTIONS)])

    return getAverageStrategy(strategy_sum),getAverageStrategy(strategy_sum_op)

# print(train_one(10000))
print(train_both(10000))

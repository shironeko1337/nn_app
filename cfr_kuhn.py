from typing import List, cast, Dict

import numpy as np

from labml import experiment, analytics
from labml.configs import option
from labml_nn.cfr import History as _History, InfoSet as _InfoSet, Action, Player, CFRConfigs
from labml_nn.cfr.infoset_saver import InfoSetSaver
from labml_nn.cfr.analytics import plot_infosets

# pass, bet
ACTIONS = cast(List[Action], ['p', 'b'])
CHANCES = cast(List[Action], ['A', 'K', 'Q'])
PLAYERS = cast(List[Player], [0, 1])


class InfoSet(_InfoSet):
    @staticmethod
    def from_dict(data: Dict[str, any]) -> 'InfoSet':
        pass
    def actions(self) -> List[Action]:
        return ACTIONS
    # probability of bet in the form of xx.x%
    def __repr__(self):
        total = sum(self.cumulative_strategy.values())
        total = max(total, 1e-6)
        bet = self.cumulative_strategy[cast(Action, 'b')] / total
        return f'{bet * 100: .1f}%'

class History(_History):
    history: str
    def __init__(self, history: str = ''):
        self.history = history

    # is a terminal state
    def is_terminal(self): # for this game, verbose, but common to other games
        if len(self.history) <= 2:
            return False
        elif self.history[-1] == 'p': # last pass
            return True
        elif self.history[-1] == 'bb': # both bets
            return True
        else:
            return False

    def _terminal_utility_p0(self) -> float:
        winner = 1 if self.history[0] < self.history[1] else -1
        if self.history[-2:] == 'bp':
            return 1
        elif self.history[-2:] == 'bb':
            return winner + 1 # 1 plus 1 blind bet
        elif self.history[-1] == 'p':
            return winner # 1 blind bet
        else:
            raise RuntimeError()
    def terminal_utility(self,i):
        if i == PLAYERS[0]:
            return self._terminal_utility_p0()
        return -self._terminal_utility_p0() # zero sum game

    # is the first two chance history
    def is_chance(self):
        return len(self.history) < 2

    # add an action to history
    def __add__(self, other:Action):
        return History(self.history + other)

    def player(self):
        return cast(Player,len(self.history) % 2)

    def sample_chance(self):
        while True:
            r = np.random.randint(len(CHANCES))
            chance = CHANCES[r]
            for c in self.history:
                if c == chance:
                    chance = None
                    break
            if chance is not None:
                return cast(Action, chance)

    def __repr__(self):
        return repr(self.history)

    # player's info = his card and all actions
    def info_set_key(self) -> str:
        i = self.player()
        return self.history[i] + self.history[2:]

    def new_info_set(self) -> InfoSet:
        return InfoSet(self.info_set_key())

def create_new_history():
    return History()

class Configs(CFRConfigs):
    pass

@option(Configs.create_new_history)
def _cnh():
  return create_new_history

def main():
    experiment.create(name='kuhn_poker',writers={'sqlite'})
    conf = Configs()
    experiment.configs(conf,{'epochs': 1_000})
    with experiment.start():
        conf.cfr.iterate()
    inds = analytics.runs(experiment.get_uuid())
    plot_infosets(inds['average_strategy.*'], width=600, height=500).display()
    analytics.scatter(inds.average_strategy_Q_b, inds.average_strategy_Kb_b,
                  width=400, height=400)

if __name__ == '__main__':
    main()

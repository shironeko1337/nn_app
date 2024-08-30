# Applications of Neural Network

## CFR demos
Python implementation of demo programs using CFR for games. For
reading the code from beginning, it's high suggested that you follow the order of RPS -> Recursive kuhn poker -> DP kuhn poker. All these demos are designed for finding
a strategy to get **close** to Nash Equalibrium in a sequential **two** player **zero sum** game.

### RPS
`cfr_rps.py` Rock-Paper-Scissors game. `train_one()` is used for train one player given another's trategy, `train_both()` is used for train both.
### Refs
- http://modelai.gettysburg.edu/2013/cfr/cfr.pdf
- Java implementation http://modelai.gettysburg.edu/2013/cfr/index.html

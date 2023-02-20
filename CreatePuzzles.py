import numpy as np

from Game import Game, randomize_hard

start_positions = []
for i in range(1, 40):
    for j in range(20):
        start = randomize_hard(i)
        start_positions.append(start.board)
start_positions = np.asarray(start_positions)
np.save("positions.npy", start_positions)
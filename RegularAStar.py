import time
import pandas as pd
import numpy as np
from Game import Solver, Game

goal = Game()
goal.set_board([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 0],
                ])

start_positions = np.load("positions.npy")
start_boards = []
for pos in start_positions:
    board = Game()
    board.set_board(pos.tolist())
    start_boards.append(board)
start_positions = start_boards

times = []
averageHeuristic = []
closedDfs = []
closedAStar = []
lengths = []


for start in start_positions:
    solver = Solver()
    startTime = time.time()
    foundPath = list(solver.astar(start, goal))
    manhattanTime = time.time() - startTime
    print(manhattanTime)
    print(len(foundPath))
    times.append(manhattanTime)
    averageHeuristic.append(solver.get_average())
    closedDfs.append(solver.dfsVisted)
    closedAStar.append(solver.closedSize)
    lengths.append(len(foundPath))

data = {'Time': times, 'Average Heuristic': averageHeuristic,
        'Expanded DFS': closedDfs, 'Expanded A*': closedAStar, 'Length': lengths}

df = pd.DataFrame(data)
df.to_csv("Regular.csv")

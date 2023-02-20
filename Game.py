import random
import numpy as np
import copy
import math
from astar import AStar
import time
from abc import ABC, abstractmethod
from heapq import heappush, heappop
from typing import Iterable, Union, TypeVar, Generic

Infinite = float("inf")
T = TypeVar("T")


class Game:

    def __init__(self):
        self.actions = [self.move_right, self.move_left, self.move_down, self.move_up]
        pass

    def stay(self):
        pass

    def empty_board(self):
        self.board = [[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0], ]
        self.update_empty()

    def set_board(self, board):
        self.board = board
        self.update_empty()

    def random_pos(self):
        return random.randrange(len(self.board)), random.randrange(len(self.board[0]))

    def randomize(self):
        self.empty_board()
        for i in range(1, len(self.board) * len(self.board[0])):
            x, y = self.random_pos()
            while self.board[x][y] != 0:
                x, y = self.random_pos()
            self.board[x][y] = i
        self.update_empty()

    def randomize_easy(self, moves):
        self.empty_board()
        for i in range(1, len(self.board) * len(self.board[0])):
            row, col = self.get_target_number_pos(i)
            self.board[row][col] = i
        self.update_empty()
        last_move = -1

        for i in range(moves):
            move = random.randint(0, len(self.actions) - 1)
            while not self.actions[move]():
                move = random.randint(0, len(self.actions) - 1)
            last_move = move

    def print_board(self):
        print("--------------")
        for row in self.board:
            print(row)

    def find_empty(self):
        for i in range(len(self.board)):
            row = self.board[i]
            for j in range(len(row)):
                if row[j] == 0:
                    return [i, j]
        return [-1, -1]

    def update_empty(self):
        self.empty = self.find_empty()

    def switch(self, x1, y1, x2, y2):
        temp = self.board[x1][y1]
        self.board[x1][y1] = self.board[x2][y2]
        self.board[x2][y2] = temp

    def move_right(self):
        if self.empty[1] >= len(self.board[0]) - 1:
            return False
        self.switch(self.empty[0], self.empty[1], self.empty[0], self.empty[1] + 1)
        self.empty[1] += 1
        return True

    def move_left(self):
        if self.empty[1] <= 0:
            return False
        self.switch(self.empty[0], self.empty[1], self.empty[0], self.empty[1] - 1)
        self.empty[1] -= 1
        return True

    def move_down(self):
        if self.empty[0] >= len(self.board) - 1:
            return False
        self.switch(self.empty[0], self.empty[1], self.empty[0] + 1, self.empty[1])
        self.empty[0] += 1
        return True

    def move_up(self):
        if self.empty[0] <= 0:
            return False
        self.switch(self.empty[0], self.empty[1], self.empty[0] - 1, self.empty[1])
        self.empty[0] -= 1
        return True

    def get_target_number_pos(self, number):
        if number == 0:
            return len(self.board) - 1, len(self.board[0]) - 1
        number -= 1
        row = int(number / len(self.board))
        col = number - row * len(self.board[0])
        return row, col

    def game_over(self):
        for i in range(1, len(self.board) * len(self.board[0])):
            row, col = self.get_target_number_pos(i)
            if self.board[row][col] != i:
                return False
        return True

    def get_man_distance(self):
        dist = 0
        for i in range(len(self.board) * len(self.board[0])):
            row = int(i / len(self.board[0]))
            col = i - row * len(self.board[0])
            number = self.board[row][col]
            target_pos = self.get_target_number_pos(number)
            dist += abs(row - target_pos[0]) + abs(col - target_pos[1])
        return dist

    def get_positions(self):
        res = dict()
        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                num = self.board[i][j]
                if num != 0:
                    res[num] = (i, j)
        return res

    def __hash__(self):
        return str(self.board).__hash__()

    def __eq__(self, other):
        if len(self.board) != len(other.board):
            return False
        if len(self.board[0]) != len(other.board[0]):
            return False

        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                if self.board[i][j] != other.board[i][j]:
                    return False

        return True


class Solver(AStar):
    def __init__(self):
        self.total = 0
        self.count = 0
        self.closedSize = 0
        self.dfsVisted = 0

    def heuristic_cost_estimate(self, game1, game2):
        val = self.manhattan(game1, game2)
        self.total += val
        self.count += 1
        return val

    def get_average(self):
        return self.total / self.count

    # def manhattan(self, game1, game2):
    # positions1 = game1.get_positions()
    # positions2 = game2.get_positions()
    # res = 0
    # for i in range(1, len(game1.board) * len(game2.board[0])):
    # x1, y1 = positions1[i]
    # x2, y2 = positions2[i]
    # res += abs(x1 - x2) + abs(y1 - y2)
    # return res
    def manhattan(self, game1, game2):
        res = 0
        for i in range(len(game1.board)):
            for j in range(len(game1.board[0])):
                num = game1.board[i][j]
                if num == 0:
                    continue
                x = (num - 1) / len(game1.board)
                y = (num - 1) % len(game1.board)
                res += abs(i - x) + abs(j - y)
        return res

    def distance_between(self, n1, n2):
        return 1

    def neighbors(self, game):
        res = []
        other = None
        for i in range(len(game.actions)):
            if other is None:
                other = create_copy(game)
            changed = other.actions[i]()
            if changed:
                res.append(other)
                other = None
        return res

    class SearchNode:
        """Representation of a search node"""

        __slots__ = ("data", "gscore", "fscore", "closed", "came_from", "out_openset", "fu", "gu", "visitedBy")

        def __init__(
                self, data: T, gscore: float = Infinite, fscore: float = Infinite
        ) -> None:
            self.data = data
            self.gscore = gscore
            self.fscore = fscore
            self.closed = False
            self.out_openset = True
            self.came_from = None
            self.fu = Infinite
            self.gu = Infinite
            self.visitedBy = -1

        def __lt__(self, b: "Solver.SearchNode") -> bool:
            return self.fscore < b.fscore

    class SearchNodeDict(dict):
        def __missing__(self, k):
            v = Solver.SearchNode(k)
            self.__setitem__(k, v)
            return v

    def astar(self, start, goal, reversePath: bool = False) -> Union[Iterable, None]:
        if self.is_goal_reached(start, goal):
            return [start]
        searchNodes = Solver.SearchNodeDict()
        startNode = searchNodes[start] = Solver.SearchNode(
            start, gscore=0.0, fscore=self.heuristic_cost_estimate(start, goal)
        )
        openSet: list = []
        heappush(openSet, startNode)
        while openSet:
            current = heappop(openSet)
            self.closedSize += 1
            if self.is_goal_reached(current.data, goal):
                return self.reconstruct_path(current, reversePath)
            current.out_openset = True
            current.closed = True
            for neighbor in map(lambda n: searchNodes[n], self.neighbors(current.data)):
                if neighbor.closed:
                    continue
                tentative_gscore = current.gscore + self.distance_between(
                    current.data, neighbor.data
                )
                if tentative_gscore >= neighbor.gscore:
                    continue
                neighbor.came_from = current
                neighbor.gscore = tentative_gscore
                neighbor.fscore = tentative_gscore + self.heuristic_cost_estimate(
                    neighbor.data, goal
                )
                if neighbor.out_openset:
                    neighbor.out_openset = False
                    heappush(openSet, neighbor)
                else:
                    # re-add the node in order to re-sort the heap
                    openSet.remove(neighbor)
                    heappush(openSet, neighbor)
        return None

    def astarLookAhead(self, start, goal, reversePath: bool = False, k=4) -> Union[Iterable, None]:
        UB = math.inf
        dfIteration = 1

        def DFS(start, LHB):
            nonlocal dfIteration
            start.gu = start.gscore
            start.fu = start.fscore
            start.visitedBy = dfIteration
            val = doDFS(start, LHB, dfIteration)
            dfIteration += 1
            return val

        def doDFS(current, LHB, it):
            nonlocal UB
            self.dfsVisted += 1
            minCost = math.inf

            fc = current.fu
            if self.is_goal_reached(current.data, goal):
                minCost = min(minCost, fc)
                UB = min(UB, fc)
            elif fc > LHB:
                minCost = min(minCost, fc)
            else:
                for neighbor in map(lambda n: searchNodes[n], self.neighbors(current.data)):
                    gn = current.gu + self.distance_between(current.data, neighbor.data)
                    if (neighbor.visitedBy != it or gn < neighbor.gu):
                        neighbor.gu = gn
                        neighbor.fu = gn + self.heuristic_cost_estimate(neighbor.data, goal)
                        neighbor.visitedBy = it
                        minCost = min(minCost, doDFS(neighbor, LHB, it))
            return minCost

        if self.is_goal_reached(start, goal):
            return [start]
        searchNodes = Solver.SearchNodeDict()
        startNode = searchNodes[start] = Solver.SearchNode(
            start, gscore=0.0, fscore=self.heuristic_cost_estimate(start, goal)
        )
        openSet: list = []
        heappush(openSet, startNode)
        while openSet:
            current = heappop(openSet)
            self.closedSize += 1
            if self.is_goal_reached(current.data, goal):
                return self.reconstruct_path(current, reversePath)
            current.out_openset = True
            current.closed = True
            for neighbor in map(lambda n: searchNodes[n], self.neighbors(current.data)):
                if neighbor.closed:
                    continue
                tentative_gscore = current.gscore + self.distance_between(
                    current.data, neighbor.data
                )
                if tentative_gscore >= neighbor.gscore:
                    continue
                neighbor.came_from = current
                neighbor.gscore = tentative_gscore
                neighbor.fscore = tentative_gscore + self.heuristic_cost_estimate(
                    neighbor.data, goal
                )
                if neighbor.fscore > UB:
                    continue
                if self.is_goal_reached(neighbor.data, goal):
                    UB = min(UB, neighbor.fscore)
                LHB = min(UB, current.fscore + k)
                if neighbor.fscore <= LHB:
                    minCost = DFS(neighbor, LHB)
                    if minCost < math.inf and minCost > neighbor.fscore:
                        neighbor.fscore = minCost

                if neighbor.out_openset:
                    neighbor.out_openset = False
                    heappush(openSet, neighbor)
                else:
                    # re-add the node in order to re-sort the heap
                    openSet.remove(neighbor)
                    heappush(openSet, neighbor)
        return None

    def astarLookAheadNew(self, start, goal, reversePath: bool = False, k=4) -> Union[Iterable, None]:
        UB = math.inf
        dfIteration = 1

        def DFS(start, LHB):
            nonlocal dfIteration
            start.gu = start.gscore
            start.fu = start.fscore
            start.visitedBy = dfIteration
            val = doDFS(start, LHB, dfIteration)
            dfIteration += 1
            return val

        def doDFS(current, LHB, it):
            nonlocal UB
            self.dfsVisted += 1
            minCost = math.inf

            fc = current.fu
            if self.is_goal_reached(current.data, goal):
                minCost = min(minCost, fc)
                UB = min(UB, fc)
            elif fc > LHB:
                minCost = min(minCost, fc)
            else:
                for neighbor in map(lambda n: searchNodes[n], self.neighbors(current.data)):
                    gn = current.gu + self.distance_between(current.data, neighbor.data)
                    if not neighbor.closed and (neighbor.visitedBy != it or gn < neighbor.gu):
                        neighbor.gu = gn
                        neighbor.fu = gn + self.heuristic_cost_estimate(neighbor.data, goal)
                        neighbor.visitedBy = it
                        minCost = min(minCost, doDFS(neighbor, LHB, it))
            return minCost

        if self.is_goal_reached(start, goal):
            return [start]
        searchNodes = Solver.SearchNodeDict()
        startNode = searchNodes[start] = Solver.SearchNode(
            start, gscore=0.0, fscore=self.heuristic_cost_estimate(start, goal)
        )
        openSet: list = []
        heappush(openSet, startNode)
        while openSet:
            current = heappop(openSet)
            self.closedSize += 1
            if self.is_goal_reached(current.data, goal):
                return self.reconstruct_path(current, reversePath)
            current.out_openset = True
            current.closed = True
            for neighbor in map(lambda n: searchNodes[n], self.neighbors(current.data)):
                if neighbor.closed:
                    continue
                tentative_gscore = current.gscore + self.distance_between(
                    current.data, neighbor.data
                )
                if tentative_gscore >= neighbor.gscore:
                    continue
                neighbor.came_from = current
                neighbor.gscore = tentative_gscore
                neighbor.fscore = tentative_gscore + self.heuristic_cost_estimate(
                    neighbor.data, goal
                )
                if neighbor.fscore > UB:
                    continue
                if self.is_goal_reached(neighbor.data, goal):
                    UB = min(UB, neighbor.fscore)
                LHB = min(UB, current.fscore + k)
                if neighbor.fscore <= LHB:
                    minCost = DFS(neighbor, LHB)
                    if minCost < math.inf and minCost > neighbor.fscore:
                        neighbor.fscore = minCost

                if neighbor.out_openset:
                    neighbor.out_openset = False
                    heappush(openSet, neighbor)
                else:
                    # re-add the node in order to re-sort the heap
                    openSet.remove(neighbor)
                    heappush(openSet, neighbor)
        return None


def create_copy(other):
    game = Game()
    game.empty = [row for row in other.empty]
    game.board = [row[:] for row in other.board]
    return game


def randomize_hard(moves):
    game = Game()
    game.empty_board()
    for i in range(1, len(game.board) * len(game.board[0])):
        row, col = game.get_target_number_pos(i)
        game.board[row][col] = i
    game.update_empty()

    already_seen = set()
    already_seen.add(str(game.board))

    count = 0
    while True:
        temp = create_copy(game)
        move = random.randint(0, len(temp.actions) - 1)
        while not temp.actions[move]():
            move = random.randint(0, len(temp.actions) - 1)
        if str(temp.board) not in already_seen:
            already_seen.add(str(temp.board))
            game = temp
            count += 1
            if count == moves:
                break
    return game

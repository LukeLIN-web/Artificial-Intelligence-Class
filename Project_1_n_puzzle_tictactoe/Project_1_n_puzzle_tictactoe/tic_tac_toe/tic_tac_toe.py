#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/2 21:06
# @Author  : yaoqi
# @Email   : yaoqi_isee@zju.edu.cn
# @File    : tic_tac_toe.py

# +++++++++++++++++++++++++++++++++++++++++++++ README ++++++++++++++++++++++++++++++++++++++++
# You will write a tic tac toc player and play game with this computer player.
# You will use MiniMaxSearch with depth limited strategy. For simplicity, Alpha-Beta Pruning
# is not considered.
# Let's make some assumptions.
# 1. The computer player(1) use circle(1) and you(-1) use cross(-1).
# 2. The computer player is MAX user and you are MIN user.
# 3. The miniMaxSearch depth is 3, so that the computer predict one step further. It first
# predicts what you will do if it makes a move and choose a move that maximize its gain.
# 4. You play first
# +++++++++++++++++++++++++++++++++++++++++++++ README ++++++++++++++++++++++++++++++++++++++++

import numpy as np
import copy


def MinimaxSearch(current_state):
    """
    Search the next step by Minimax Search with depth limited strategy
    The search depth is limited to 3, computer player(1) uses circle(1) and you(-1) use cross(-1)
    :param current_state: current state of the game, it's a 3x3 array representing the chess board, array element lies
    in {1, -1, 0}, standing for circle, cross, empty place. 电脑下1, 我们下-1, 空为0
    :return: row and column index that computer player will draw a circle on. Note 0<=row<=2, 0<=col<=2
    """
    # -------------------------------- Your code starts here ----------------------------- #
    assert isinstance(current_state, np.ndarray)  # 实例化一个n dimension array
    assert current_state.shape == (3, 3)

    # get available actions 它自带有copy功能
    game_state = current_state.copy()
    actions = get_available_actions(game_state)

    # computer player: traverse all the possible actions and maximize the utility function
    values = []
    depth = 3
    for action in actions:
        acmin = min_value(action_result(game_state.copy(), action, 1), depth)
        values.append(acmin)
    max_ind = int(np.argmax(values))  # 找出最大值
    row, col = actions[max_ind][0], actions[max_ind][1]  # 返回电脑下的地方

    return row, col


def get_available_actions(current_state):
    """
    get all the available actions given current state
    :param current_state:current state of the game, it's a 3x3 array
    :return: available actions. list of tuple [(r0, c0), (r1, c1), (r2, c2)]
    """
    assert isinstance(current_state, np.ndarray), 'current_state should be numpy ndarray'
    assert current_state.shape == (3, 3), 'current_state: expect 3x3 array, get {}'.format(current_state.shape)
    res = []
    for i in range(len(current_state)):
        for j in range(len(current_state[i])):
            #      print(current_state[i][j])  # for debug
            if current_state[i][j] == 0:
                xy = (i, j)
                res.append(xy)
    return res


def action_result(current_state, action, player):
    """
    update the game state given the input action and player
    :param current_state: current state of the game, it's a 3x3 array
    :param action: current action, tuple type
    :param player:
    :return: current_state
    """
    assert isinstance(current_state, np.ndarray), 'current_state should be numpy ndarray'
    # judge whether or not cur state is a ndarray
    assert current_state.shape == (3, 3), 'current_state: expect 3x3 array, get {}'.format(current_state.shape)
    assert player in [1, -1], 'player should be either 1(computer) or -1(you)'
    #  lines above check the input is correct
    current_state[action[0]][action[1]] = player

    return current_state


# 用法:  min_value(action_result(game_state.copy(), action, 1), depth)
def min_value(current_state, depth):
    """
    recursively call min_value and max_value, min_value is for human player(-1)
    :param current_state:
    :param depth:
    :return:a utility value
    """
    g = GameJudge()
    g.game_state = current_state
    if g.check_game_status() != 2 or depth == 0:  # terminate
        return utility(current_state, 1)
    v = 1000000
    s: tuple
    for s in get_available_actions(current_state):
        # print("before" + str(current_state[0][1]))
        tmpstate = action_result(copy.deepcopy(current_state), s, -1)
        m = max_value(tmpstate, depth - 1)  # 把每个可以用的action都试一遍
        # print("after" + str(current_state[0][1]))
        if m < v:  # because always computer use this function ,so player = 1
            v = m
    return v


def max_value(current_state, depth):
    """
    recursively call min_value and max_value, max_value is for computer(1)
    :param current_state:
    :param depth:
    :return: a utility value
    """
    g = GameJudge()
    g.game_state = current_state
    if g.check_game_status() != 2 or depth == 0:  # terminate
        return utility(current_state, 1)
    v = -1000000
    for s in get_available_actions(current_state):
        tmpstate = action_result(copy.deepcopy(current_state), s, 1)
        m = min_value(tmpstate, depth - 1)
        if m > v:  # because always computer use this function ,so player = 1
            v = m
    return v


# flag 有啥用?
# 电脑玩家是MAX user，你是MIN user，也就是说电脑的落子要使utility最大，而你落子要使 utility最小；
def utility(current_state, flag):
    """
    return utility function given current state and flag
    :param current_state:
    :param flag:
    :return:
    """
    g = GameJudge()
    g.game_state = current_state
    if g.check_game_status() == 1:
        return 1
    if g.check_game_status() == -1:
        return -1
    if g.check_game_status() == 0:
        return 0
    else:  # g.check_game_status() == 0: 这里应该是 判断未完成的时候的.
        uti = 0
        sum_rows = np.sum(g.game_state, axis=1).tolist()  # axis=1按行的方向相加，返回每个行的值；axis=0按列相加，返回每个列的值。
        sum_cols = np.sum(g.game_state, axis=0).tolist()
        diag = []
        diag += g.game_state[0][0] + g.game_state[1][1] + g.game_state[2][2]
        rdiag = []
        rdiag += g.game_state[0][2] + g.game_state[1][1] + g.game_state[2][0]
        # 行
        for i in range(3):
            if sum_rows[i] == 1:
                if -1 not in g.game_state[i, :]:
                    uti += 1  # 如果有-1 那就是0 , 如果没有就是 一个1
            if sum_rows[i] == 2:
                uti += 3
            if sum_rows[i] == -1:
                if 1 not in g.game_state[i, :]:
                    uti -= 1
            if sum_rows[i] == -2:
                uti -= 3
        # 列
        for i in range(3):
            if sum_cols[i] == 1:
                if -1 not in g.game_state[:, i]:
                    uti += 1
            if sum_cols[i] == 2:
                uti += 3
            if sum_cols[i] == -1:
                if 1 not in g.game_state[:, i]:
                    uti -= 1
            if sum_cols[i] == -2:
                uti -= 3
        # 对角线
        for i in range(3):
            if sum(diag) == 1:
                if -1 not in diag:
                    uti += 1
            if sum(diag) == 2:
                uti += 3
            if sum(diag) == -1:
                if 1 not in diag:
                    uti -= 1
            if sum(diag) == -2:
                uti -= 3
        # 另一条对角线
        for i in range(3):
            if sum(rdiag) == 1:
                if -1 not in rdiag:
                    uti += 1
            if sum(rdiag) == 2:
                uti += 3
            if sum(rdiag) == -1:
                if 1 not in rdiag:
                    uti -= 1
            if sum(rdiag) == -2:
                uti -= 3
        return uti


# Do not modify the following code using for judge win or lose
class GameJudge(object):
    def __init__(self):
        self.game_state = np.zeros(shape=(3, 3), dtype=int)

    # 初始化为0
    def make_one_move(self, row, col, player):
        """
        make one move forward
        :param row: row index of the circle(cross)
        :param col: column index of the circle(cross)
        :param player: player = 1 for computer / player = -1 for human
        :return:
        """
        # 1 stands for circle, -1 stands for cross, 0 stands for empty
        assert 0 <= row <= 2, "row index of the move should lie in [0, 2]"
        assert 0 <= col <= 2, "column index of the move should lie in [0, 2]"
        assert player in [-1, 1], "player should be noted as -1(human) or 1(computer)"
        self.game_state[row, col] = player

    def check_game_status(self):
        """
        return game status
        :return: 1 for computer wins, -1 for human wins, 0 for draw平局, 2 in the play
        """

        # somebody wins
        sum_rows = np.sum(self.game_state, axis=1).tolist()
        if 3 in sum_rows:
            return 1
        if -3 in sum_rows:
            return -1

        sum_cols = np.sum(self.game_state, axis=0).tolist()
        if 3 in sum_cols:
            return 1
        if -3 in sum_cols:
            return -1

        sum_diag = self.game_state[0][0] + self.game_state[1][1] + self.game_state[2][2]
        if sum_diag == 3:
            return 1
        if sum_diag == -3:
            return -1

        sum_rdiag = self.game_state[0][2] + self.game_state[1][1] + self.game_state[2][0]
        if sum_rdiag == 3:
            return 1
        if sum_rdiag == -3:
            return -1

        # draw np.where(condition) 当where内只有一个参数时，那个参数表示条件，当条件成立时，where返回的是每个符合condition条件元素的坐标,返回的是以元组的形式
        if len(np.where(self.game_state == 0)[0]) == 0:
            return 0

        # in the play
        return 2

    def human_input(self):
        """
        take the human's move in
        :return: row and column index of human input
        """
        print("Input the row and column index of your move")
        print("1, 0 means draw a cross on the row 1, col 0")
        ind, succ = None, False
        while not succ:
            ind = list(map(int, input().strip().split(',')))
            if ind[0] < 0 or ind[0] > 2 or ind[1] < 0 or ind[1] > 2:
                succ = False
                print("Invalid input, the two numbers should lie in [0, 2]")
            elif self.game_state[ind[0], ind[1]] != 0:
                succ = False
                print(" You can not put cross on places already occupied")
            else:
                succ = True
        return ind[0], ind[1]

    def print_status(self, player, status):
        """
        print the game status
        :param player: player of the last move
        :param status: game status
        :return:
        """
        print("-----------------------------------------------------")
        for row in range(3):
            for col in range(3):
                if self.game_state[row, col] == 1:
                    print("[O]", end="")
                elif self.game_state[row, col] == -1:
                    print("[X]", end="")
                else:
                    print("[ ]", end="")
            print("")

        if player == 1:
            print("Last move was conducted by computer")
        elif player == -1:
            print("Last move was conducted by you")

        if status == 1:
            print("Computer wins")
        elif status == -1:
            print("You win")
        elif status == 2:
            print("Game going on")
        elif status == 0:
            print("Draw")

    def get_game_state(self):
        return self.game_state

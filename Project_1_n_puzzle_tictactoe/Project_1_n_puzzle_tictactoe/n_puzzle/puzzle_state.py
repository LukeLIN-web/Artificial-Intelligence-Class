from typing import List, Any

import numpy as np
from enum import Enum
import copy
import heapq


# Enum of operation in EightPuzzle problem
class Move(Enum):
    """
    The class of move operation
    NOTICE: The direction denotes the 'blank' space move
    """
    Up = 0
    Down = 1
    Left = 2
    Right = 3


# EightPuzzle state
class PuzzleState(object):
    """
    Class for state in EightPuzzle-Problem , show problem state
    Attr:
        square_size: Chessboard size, e.g: In 8-puzzle problem, square_size = 3
        state: 'square_size' x 'square_size square', '-1' indicates the 'blank' block
        (For 8-puzzle, state is a 3 x 3 array)
        g: The cost from initial state to current state
        h: The value of heuristic function
        pre_move:  The previous operation to get to current state
        pre_state: Parent state of this state
    """
    square_size: int

    def __init__(self, square_size=4):
        self.square_size = square_size
        self.state = None
        self.g = 0
        self.h = 0
        self.pre_move = None
        self.pre_state = None

        self.generate_state()

    # 重写了__eq__()方法的类会隐式的把__hash__赋为None
    def __eq__(self, other):
        return (self.state == other.state).all()

    # I override lt for heap queue
    def __lt__(self, other):
        return self.g + self.h < other.g + other.h  # 小

    def blank_pos(self):
        """
        Find the 'blank' position of current state
        :return:
            row: 'blank' row index, '-1' indicates the current state may be invalid
            col: 'blank' col index, '-1' indicates the current state may be invalid
        """
        index = np.argwhere(self.state == -1)
        row = -1
        col = -1
        if index.shape[0] == 1:  # find blank
            row = index[0][0]
            col = index[0][1]
        return row, col

    def num_pos(self, num):
        """
        Find the 'num' position of current state
        :return:
            row: 'num' row index, '-1' indicates the current state may be invalid
            col: 'num' col index, '-1' indicates the current state may be invalid
        """
        index = np.argwhere(self.state == num)
        row = -1
        col = -1
        if index.shape[0] == 1:  # find number
            row = index[0][0]
            col = index[0][1]
        return row, col

    def is_valid(self):
        """
        Check current state is valid or not (A valid state should have only one 'blank')
        :return:
            flag: boolean, True - valid state, False - invalid state
        """
        row, col = self.blank_pos()
        if row == -1 or col == -1:
            return False
        else:
            return True

    def clone(self):
        """
        Return the state's deepcopy
        :return:
        """
        return copy.deepcopy(self)

    def generate_state(self, random=False, seed=None):
        """
        Generate a new state 你可以通过调用 generate_state()成员函数来随机生成棋盘格的状态,**需要设定random参数**
        :param random: True - generate state randomly, False - generate a normal state
        :param seed: Choose the seed of random, only used when random = True
        :return:
        """
        self.state = np.arange(0, self.square_size ** 2).reshape(self.square_size, -1)
        self.state[self.state == 0] = -1  # Set blank

        if random:
            np.random.seed(seed)
            np.random.shuffle(self.state)

    def display(self):
        """
        Print state
        :return:
        """
        print("----------------------")
        for i in range(self.state.shape[0]):  # tuple of array dimensions"
            # print("{}\t{}\t{}\t".format(self.state[i][0], self.state[i][1], self.state[i][2]))
            # print(self.state[i, :])
            for j in range(self.state.shape[1]):
                if j == self.state.shape[1] - 1:
                    print("{}\t".format(self.state[i][j]))
                else:
                    print("{}\t".format(self.state[i][j]), end='')
        print("----------------------\n")


def check_move(curr_state, move):
    """
    Check the operation 'move' can be performed on current state 'curr_state'
    :param curr_state: Current puzzle state
    :param move: Operation to be performed
    :return:
        valid_op: boolean, True - move is valid; False - move is invalid
        src_row: int, current blank row index
        src_col: int, current blank col index
        dst_row: int, future blank row index after move
        dst_col: int, future blank col index after move
    """
    # assert isinstance(move, Move)  # Check operation type
    assert curr_state.is_valid()

    if not isinstance(move, Move):
        move = Move(move)

    src_row, src_col = curr_state.blank_pos()
    dst_row, dst_col = src_row, src_col
    # valid_op = False

    if move == Move.Up:  # Number moves up, blank moves down
        dst_row -= 1
    elif move == Move.Down:
        dst_row += 1
    elif move == Move.Left:
        dst_col -= 1
    elif move == Move.Right:
        dst_col += 1
    else:  # Invalid operation
        dst_row = -1
        dst_col = -1

    if dst_row < 0 or dst_row > curr_state.state.shape[0] - 1 or dst_col < 0 or dst_col > curr_state.state.shape[1] - 1:
        valid_op = False
    else:
        valid_op = True

    return valid_op, src_row, src_col, dst_row, dst_col


def once_move(curr_state, move):
    """
    Perform once move to current state
    :param curr_state:
    :param move:
    :return:
        valid_op: boolean, flag of this move is valid or not. True - valid move, False - invalid move
        next_state: EightPuzzleState, state after this move
    """
    curr_state: PuzzleState
    valid_op, src_row, src_col, dst_row, dst_col = check_move(curr_state, move)

    next_state = curr_state.clone()

    if valid_op:
        it = next_state.state[dst_row][dst_col]
        next_state.state[dst_row][dst_col] = -1
        next_state.state[src_row][src_col] = it
        next_state.pre_state = curr_state
        next_state.pre_move = move
        return True, next_state
    else:
        return False, next_state
    # fail op will return curr_state copy


def check_state(src_state, dst_state):
    """
    Check current state is same as destination state
    :param src_state:
    :param dst_state:
    :return:
    """
    return (src_state.state == dst_state.state).all()


def run_moves(curr_state, dst_state, moves):
    """
    Perform list of move to current state, and check the final state is same as destination state or not
    Ideally, after we perform moves to current state, we will get a state same as the 'dst_state'
    :param curr_state: EightPuzzleState, current state
    :param dst_state: EightPuzzleState, destination state
    :param moves: List of Move用于判断生成的移动指令是否可以从初始状态到达目标状态
    :return:
        flag of moves: True - We can get 'dst_state' from 'curr_state' by 'moves'
    """
    pre_state = curr_state.clone()
    next_state = None

    for move in moves:
        valid_move, next_state = once_move(pre_state, move)

        if not valid_move:
            return False

        pre_state = next_state.clone()

    if check_state(next_state, dst_state):
        return True
    else:
        return False


def runs(curr_state, moves):
    """
    Perform list of move to current state, get the result state
    NOTICE: The invalid move operation would be ignored
    :param curr_state:
    :param moves:
    :return:
    """
    pre_state = curr_state.clone()
    next_state = None

    for move in moves:
        valid_move, next_state = once_move(pre_state, move)
        pre_state = next_state.clone()
    return next_state


def print_moves(init_state, moves):
    """
    While performing the list of move to current state, this function will also print how each move is performed
    :param init_state: The initial state
    :param moves: List of move
    :return: 会打印中间移动指令的执行过程
    """
    print("Initial state")
    init_state.display()

    pre_state = init_state.clone()
    next_state = None

    for idx, move in enumerate(moves):
        if move == Move.Up:  # Number moves up, blank moves down
            print("{} th move. Goes up.".format(idx))
        elif move == Move.Down:
            print("{} th move. Goes down.".format(idx))
        elif move == Move.Left:
            print("{} th move. Goes left.".format(idx))
        elif move == Move.Right:
            print("{} th move. Goes right.".format(idx))
        else:  # Invalid operation
            print("{} th move. Invalid move: {}".format(idx, move))

        valid_move, next_state = once_move(pre_state, move)

        if not valid_move:
            print("Invalid move: {}, ignore".format(move))

        next_state.display()

        pre_state = next_state.clone()

    print("We get final state: ")
    next_state.display()


def generate_moves(move_num=30):
    """
    Generate a list of move in a determined length randomly
    :param move_num:
    :return:
        move_list: list of move
    """
    move_dict = {}
    move_dict[0] = Move.Up
    move_dict[1] = Move.Down
    move_dict[2] = Move.Left
    move_dict[3] = Move.Right

    index_arr = np.random.randint(0, 4, move_num)
    index_list = list(index_arr)

    move_list = [move_dict[idx] for idx in index_list]

    return move_list


def convert_moves(moves):
    """
    Convert moves from int into Move type
    :param moves:
    :return:
    """
    if len(moves):
        if isinstance(moves[0], Move):
            return moves
        else:
            return [Move(move) for move in moves]
    else:
        return moves


def update_cost(child_state, dst_state):
    child_state: PuzzleState
    dst_state: PuzzleState
    child_state.g = child_state.pre_state.g + 1
    manhattan = 0
    for i in range(1, child_state.square_size ** 2):
        dst_row, dst_col = dst_state.num_pos(i)
        chi_row, chi_col = child_state.num_pos(i)
        manhattan += abs(dst_row - chi_row) + abs(dst_col - chi_col)
    child_state.h = manhattan  # find the distance between child's blank and dst's blank
    return child_state


# 返回是否在list中
def state_in_list(child_state, argList):
    argList: List
    for it in argList:
        if child_state.__eq__(it):
            return True, it
    return False, None


def expand_state(curr_state):
    childs: List[Any] = []
    # find the block location , if  x> 0 ,
    row, col = curr_state.blank_pos()
    # block can move up  前row -1行,
    if row > 0:
        flag, s1 = once_move(curr_state, Move.Up)
        if flag is True:
            childs.append(s1)

    # block can move down    Up = 0
    #     Down = 1
    #     Left = 2
    #     Right = 3
    if row < curr_state.square_size - 1:
        flag, s2 = once_move(curr_state, Move.Down)
        if flag is True:
            childs.append(s2)

    # block can move left
    if col > 0:
        flag, s3 = once_move(curr_state, Move.Left)
        if flag is True:
            childs.append(s3)

    # block can move right
    if col < curr_state.square_size - 1:  # col < curr_state.square_size是不行的, 这样 3到最右边了, 但是还是会进入if语句
        flag, s4 = once_move(curr_state, Move.Right)
        if flag is True:
            childs.append(s4)
    return childs


def get_path(curr_stat):
    res = []
    while curr_stat.pre_state is not None:
        res.append(curr_stat.pre_move)
        curr_stat = curr_stat.pre_state
    res.reverse()
    return res


"""
NOTICE:
1. init_state is a 3x3 numpy array, the "space" is indicated as -1, for example
    1 2 -1              1 2
    3 4 5   stands for  3 4 5
    6 7 8               6 7 8
2. moves contains directions that transform initial state to final state. Here
    0 stands for up
    1 stands for down
    2 stands for left
    3 stands for right
    
   There might be several ways to understand "moving up/down/left/right". Here we define
   that "moving up" means to move 'space' up, not move other numbers up. For example
    1 2 5                1 2 -1
    3 4 -1   move up =>  3 4 5
    6 7 8                6 7 8
   This definition is actually consistent with where your finger moves to
   when you are playing 8 puzzle game.
   需要自己实现,函数输入参数为init_state（起始状态）与dst_state（目标状态），返回值为move_list（移动指令列表）。
   what's more, there is no other function use astar 
3. It's just a simple example of A-Star search. You can implement this function in your own design.  
"""


def astar_search_for_puzzle_problem(init_state, dst_state):
    """
    Use AStar-search to find the path from init_state to dst_state
    init_state and dst state are  PuzzleState classes
    :param init_state:  Initial puzzle state
    :param dst_state:   Destination puzzle state
    :return:  All operations needed to be performed from init_state to dst_state
        moves: list of Move. e.g: move_list = [Move.Up, Move.Left, Move.Right, Move.Up]
    """

    start_state = init_state.clone()
    # end_state = dst_state.clone()

    open_list = []  # I use priority queue instead of list, List[PuzzleState]
    close_list = []  # close list is a list[PuzzleState]

    # move_list = []  # The operations from init_state to dst_state

    # Initial A-star
    heapq.heappush(open_list, start_state)

    while len(open_list) > 0:
        # Get best node from open_list
        # you can define list insertion function to implement priority deque to improve the efficiency of algorithm
        curr_state = open_list[0]  # before line is :curr_idx, curr_state = find_front_node(open_list)

        # Delete best node from open_list
        heapq.heappop(open_list)  # open_list.pop(curr_idx) 这样要找到再pop，O（n）比较慢

        # Add best node in close_list
        close_list.append(curr_state)

        # Check whether found solution， return the path
        moves: list
        if curr_state == dst_state:
            moves = get_path(curr_state)
            return moves

        # Expand node
        childs = expand_state(curr_state)

        # calculate the fn of each node state
        for child_state in childs:
            # Explored node, if this state in the close_list, do not consider it.
            in_list, match_state = state_in_list(child_state, close_list)
            # if this state in close list, we don't judge it .
            if in_list:
                continue

            # Assign cost(including g and h) to child state. You can also do this in Expand operation
            child_state = update_cost(child_state, dst_state)

            # Find a better state in open_list 如果找到一样的就没必要加入了.
            in_list, match_state = state_in_list(child_state, open_list)
            if in_list:
                continue

            # if open_list.__len__() > 0:
            #     fno = open_list[0].g + open_list[0].h
            #     fnc = child_state.g + child_state.h
            #     if fnc < fno:
            #         heapq.heappush(open_list, child_state)
            # else:
            heapq.heappush(open_list, child_state)
            # open_list.append(child_state)

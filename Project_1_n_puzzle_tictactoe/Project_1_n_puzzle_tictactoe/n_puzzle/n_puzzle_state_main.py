from puzzle_state import PuzzleState, astar_search_for_puzzle_problem, run_moves, generate_moves, print_moves, \
    convert_moves, runs
import numpy as np


def main():
    # Create a initial state randomly
    square_size = 4

    init_state = PuzzleState(square_size=square_size)
    # dst = [1, 2, 3,
    #        8,-1, 6,
    #        7, 4, 5]
    dst = [1, 4, 5, 14,
           2, 6, 13, 15,
           11, 7, -1, 10,
           8, 9, 12, 3]
    init_state.state = np.asarray(dst).reshape(square_size, square_size)

    move_list = generate_moves(100)  # 起始状态是通过在目标状态上随机执行一定步数的移动指令生成，当前设置为100步
    init_state.state = runs(init_state, move_list).state

    # Set a determined destination state
    dst_state = PuzzleState(square_size=square_size)
    dst_state.state = np.asarray(dst).reshape(square_size, square_size)

    # Find the path from 'init_state' to 'dst_state'
    move_list = astar_search_for_puzzle_problem(init_state, dst_state)

    move_list = convert_moves(move_list)

    # Perform your path
    if run_moves(init_state, dst_state, move_list):
        print_moves(init_state, move_list)
        print("Our dst state: ")
        dst_state.display()
        print("Get to dst state. Success !!!")
    else:
        print_moves(init_state, move_list)
        print("Our dst state: ")
        dst_state.display()
        print("Can not get to dst state. Failed !!!")


main()

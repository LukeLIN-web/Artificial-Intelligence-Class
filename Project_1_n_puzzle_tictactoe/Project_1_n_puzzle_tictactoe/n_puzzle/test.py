import numpy

from puzzle_state import PuzzleState

row = 1
col = 0
# lit = curr_state[:row]+[curr_state[row][:col]+curr_state[row][col+1:col+2]+curr_state[row][col:col+1]+curr_state[row][col+2:]]+curr_state[row+1:]
# print(curr_state)
# print(lit)
curr_state = PuzzleState()
curr_state.state = numpy.ones(shape=(4, 4))

s1 = PuzzleState()
s1.state = numpy.zeros(shape=(4, 4))
s1.state = curr_state.state[:row - 1] + [
    curr_state.state[row - 1][:col] + curr_state.state[row][col:col + 1] + curr_state.state[row - 1][col + 1:]] + [
               curr_state.state[row][:col] + curr_state.state[row - 1][col:col + 1] + curr_state.state[row][
                                                                                      col + 1:]] + curr_state.state[
                                                                                                   row + 1:]
print(s1.state)

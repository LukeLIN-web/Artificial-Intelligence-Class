
curr_state = [[1,2], [3,4], [5,6]]
row = 1
col =0
lit = curr_state[:row]+[curr_state[row][:col]+curr_state[row][col+1:col+2]+curr_state[row][col:col+1]+curr_state[row][col+2:]]+curr_state[row+1:]
print(curr_state)
print(lit)

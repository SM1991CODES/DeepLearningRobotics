import numpy as np
# ----------
# User Instructions:
# 
# Create a function compute_value which returns
# a grid of values. The value of a cell is the minimum
# number of moves required to get from the cell to the goal. 
#
# If a cell is a wall or it is impossible to reach the goal from a cell,
# assign that cell a value of 99.
# ----------

grid = [[0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0]]
goal = [len(grid)-1, len(grid[0])-1]
cost = 1 # the cost associated with moving from a cell to an adjacent one

delta = [[-1, 0 ], # go up
         [ 0, -1], # go left
         [ 1, 0 ], # go down
         [ 0, 1 ]] # go right

delta_name = ['^', '<', 'v', '>']

def compute_value(grid,goal,cost):
    # ----------------------------------------
    # insert code below
    # ----------------------------------------

    r, c = np.where(grid == 1)
    grid[r, c] = 999  # fill all non-navigable spaces with this
    r, c = np.where(grid != 999)
    grid[r, c] = 99  # fill all other cells with 99
    grid[goal[0], goal[1]] = 0  # goal costs 0

    print("init grid\n", grid)

    # now do the following till we have cells with value 99
    r, c = np.where(grid == 99)

    iter = 0
    while  len(r) > 0:

      for i in range(0, grid.shape[0]):
        for j in range(0, grid.shape[1]):

          # check if a cell has value 99
          if grid[i, j] == 99:
            adjacent_cell_vals = []

            # look for valid values in it's 4 possible adjacent cells
            if i - 1 >= 0 and grid[i - 1, j] != 999 and grid[i - 1, j] != 99:
              adjacent_cell_vals.append(grid[i - 1, j])
            
            if i + 1 < grid.shape[0] and grid[i + 1, j] != 999 and grid[i + 1, j] != 99:
              adjacent_cell_vals.append(grid[i + 1, j])
            
            if j - 1 >= 0 and grid[i, j - 1] != 999 and grid[i, j - 1] != 99:
              adjacent_cell_vals.append(grid[i, j - 1])

            if j + 1 < grid.shape[1] and grid[i, j + 1] != 999 and grid[i, j + 1] != 99:
              adjacent_cell_vals.append(grid[i, j + 1])

            # if we have some values, select min and add cost
            if len(adjacent_cell_vals) > 0:
              grid[i, j] = np.min(adjacent_cell_vals) + cost
      
      iter += 1
      print("After iteration {0} \n".format(iter))
      print(grid)
      r, c = np.where(grid == 99)

      if iter > grid.shape[0] * grid.shape[1]:
          break

    # make sure your function returns a grid of values as 
    # demonstrated in the previous video.
    return grid 


if __name__ == '__main__':

  print(goal)
  value_grid = compute_value(grid=np.array(grid), goal=goal, cost=cost)

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
        [0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 0]]
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
      # print("After iteration {0} \n".format(iter))
      # print(grid)
      r, c = np.where(grid == 99)

      if iter > grid.shape[0] * grid.shape[1]:
          break

    # make sure your function returns a grid of values as 
    # demonstrated in the previous video.
    return grid 


def get_least_cost_path_to_goal(start_loc, goal_loc, value_grid):
  """
  Function returns a char matrix with least cost path from start to goal given the procomputed value grid
  """

  present_cell_index = start_cell

  path_grid = np.chararray(value_grid.shape, itemsize=5)
  path_grid = value_grid.copy().astype(object)
  print("Value grid as char array\n")
  print(path_grid)

  itercount = 0

  while present_cell_index != goal_loc:

    # get the 4 neighbours of present cell and get the minimum cost cell
    adjacent_vals = [999, 999, 999, 999]  # default values, in the order top, bottom, left, right
    directions = ['^', 'v', '<', '>']

    rT, cT = present_cell_index[0] - 1, present_cell_index[1]  # top cell
    if rT >= 0:
      adjacent_vals[0] = value_grid[rT, cT]
    
    rB, cB = present_cell_index[0] + 1, present_cell_index[1]  # bottom cell
    if rB < value_grid.shape[1]:
      adjacent_vals[1] = value_grid[rB, cB]

    rL, cL = present_cell_index[0], present_cell_index[1] - 1  # left cell
    if cL >= 0:
      adjacent_vals[2] = value_grid[rL, cL]

    rR, cR = present_cell_index[0], present_cell_index[1] + 1  # right cell
    if cR < value_grid.shape[1]:
      adjacent_vals[3] = value_grid[rR, cR]
    
    # now get the least value and get the index
    min_val = np.argmin(adjacent_vals)

    if min_val == 0:
      path_grid[rT, cT] =  directions[0] + str(path_grid[rT, cT])
      present_cell_index = (rT, cT)
    elif min_val == 1:
      path_grid[rB, cB] = directions[1] + str(path_grid[rB, cB]) 
      present_cell_index = (rB, cB)
    elif min_val == 2:
      path_grid[rL, cL] = directions[2] + str(path_grid[rL, cL])
      present_cell_index = (rL, cL)
    elif min_val == 3:
      path_grid[rR, cR] = directions[3] + str(path_grid[rR, cR])
      present_cell_index = (rR, cR)

    # print("After iter --> {0}\n".format(itercount))
    # print(path_grid)

    # continue
  
  return path_grid

if __name__ == '__main__':

  print(goal)
  value_grid = compute_value(grid=np.array(grid), goal=goal, cost=cost)

  r, c = np.where(value_grid == 999)
  value_grid[r, c] = 99

  print(" value grid \n")
  print(value_grid)

  # now given the value grid, compute least cost path for any given start location
  start_cell = (1, 0)

  path = get_least_cost_path_to_goal(start_cell, goal_loc=(4, 5), value_grid=value_grid)
  print('\n')
  print(np.array(grid))
  print(path)

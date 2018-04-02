import numpy as np
import matplotlib.pyplot as plt


class Maze(object):
    """
    For each cell in the maze, there is a unique path to every other
    cell in such a way as to never pass through any cell twice.
    """

    DIRECTIONS = ('E', 'S', 'W', 'N')

    def __init__(self, 
                 n_rows=10, 
                 n_cols=10, 
                 seed=None, 
                 start_cell=(0, 0),
                 end_cell=None,
                 show_construction=False):
        """
        Create an instance of a Maze object.
        
        Args:
            n_rows (int): the number of rows
            n_cols (int): the number of columns
            seed (int): 
            start_cell (int, int): zero-indexed row, column pair  
                indicating the starting point of the maze; default
                value of (0,0)
            end_cell (int, int): zero-indexed row, column pair
                indicating the end of the maze; if no value is given
                the end will be the lower-right corner of
                (n_rows - 1, n_cols - 1)
            show_construction (bool): whether or not to display the
                construction of the maze
        """
        # TODO: replace show_construction parameter with a method
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.shape = (n_rows, n_cols)
        
        self.seed = seed
        
        self.start_cell = start_cell
        # If no ending cell for the maze is specified, use
        # the lower-right corner
        if end_cell is None:
            self.end_cell = (n_rows - 1, n_cols - 1)
        else:
            self.end_cell = end_cell
            
        self.solution = None
        
        # Get array index of starting and ending cells
        self.start_index = self.start_cell[0]*n_cols + self.start_cell[1]
        self.end_index = self.end_cell[0]*n_cols + self.end_cell[1]
        
        # Set the random seed if provided to allow reproducibility
        np.random.seed(self.seed)
        
        # Initialize array for union-find object
        # Each maze cell starts out as its own connected component
        maze_tree = [- 1 for _ in range(self.n_rows * self.n_cols)]

        # Construct list of (interior) walls
        potential_walls = ([(j + self.n_cols*i, (j + 1) + self.n_cols*i) 
                            for i in range(self.n_rows) 
                            for j in range(self.n_cols - 1)]
                           + [(j + self.n_cols*i, j + self.n_cols*(i + 1))
                              for j in range(self.n_cols) 
                              for i in range(self.n_rows - 1)])
        
        # Initialize list of walls contained in the built maze
        self.maze_walls = []

        if show_construction:
            old_walls = self.maze_walls
            self.maze_walls = potential_walls
            print('Starting with all possible walls, each maze cell is its own',
                  ' connected component.')
            print('\nUnion-Find Array:\n{}'.format(maze_tree))
            plt.figure(figsize=[0.6*n_cols, 0.6*n_rows])
            self.plot()
            for i in range(n_cols):
                for j in range(n_rows):
                    plt.text(i, j, i + n_cols * j, fontsize=14,
                             horizontalalignment='center',
                             verticalalignment='center')
            plt.show()
            self.maze_walls = old_walls

        # Decide whether each wall should stay or be discarded
        while len(potential_walls) > 0:
            # Randomly select wall
            random_wall_index = np.random.randint(len(potential_walls))
            random_wall = potential_walls.pop(random_wall_index)

            if show_construction:
                print('Next Wall: {}'.format(random_wall))

            # Determine the connected components of the cells on either
            # side of the wall
            component_0 = find_component(maze_tree, random_wall[0])
            component_1 = find_component(maze_tree, random_wall[1])
            
            # Remove walls separating different components and keep
            # the rest
            if component_0 == component_1:
                message = ('Cells {} and {} are already connected -> keep wall!'
                           .format(random_wall[0], random_wall[1]))
                self.maze_walls.append(random_wall)
            else:
                message = ('Cells {} and {} are not connected --> remove wall!'
                           .format(random_wall[0], random_wall[1]))
                maze_tree[component_1] = component_0

            if show_construction:
                print(message)
                print('')

                if component_0 != component_1:
                    print('Updated Union-Find Array:\n{}'.format(maze_tree))

                    old_walls = self.maze_walls
                    self.maze_walls = potential_walls + old_walls

                    plt.figure(figsize=[0.6*n_cols, 0.6*n_rows])
                    self.plot()
                    for i in range(n_cols):
                        for j in range(n_rows):
                            plt.text(i, j, i + n_cols * j, fontsize=14,
                                     horizontalalignment='center',
                                     verticalalignment='center')
                    plt.show()

                    self.maze_walls = old_walls
                
    def shape(self):
        """
        Returns the shape of the maze.
        
        Returns:
            shape (int, int) : the number of rows followed by the 
                number of columns
        """
        shape = (self.n_rows, self.n_cols)
        return shape
    
    def walls(self):
        """
        Returns the list of (interior) walls from which the maze is 
        built.
        
        Returns:
            maze_walls (list): list of tuples (cell1, cell2) denoting 
                the wall separating cell1 from cell2
        """
        return self.maze_walls
    
    def get_seed(self):
        """
        Returns the random seed used to generate the maze
        
        Returns:
            seed (int): the random seed used to generate the maze
        """
        return self.seed
    
    def plot(self, 
             with_solution=False, 
             solution_rgb=(1, 1, 0),
             solution_alpha=0.75):
        """
        Args:
            with_solution (bool): whether or not to plot the maze 
                solution
            solution_rgb (int, int, int): RGB color values used to
                draw the solution if with_solution True
            solution_alpha (float): opacity of maze solution if 
                with_solution is True
        """
        # Compute centers of maze cells separated by walls
        points = [((p[0] % self.n_cols, p[1] % self.n_cols),
                   (p[0] // self.n_cols, p[1] // self.n_cols))
                  for p in self.maze_walls]
        
        # Set x- and y-axis ranges to include the full maze
        plt.xlim([-0.5, self.n_cols - 0.5])
        plt.ylim([self.n_rows - 0.5, -0.5])
        
        # Plot each maze wall
        for p in points:
            # Get endpoints of line segment forming the wall
            q = rotate_line(p)
            # Plot the wall
            plt.plot(q[0], q[1], c='black')
        
        # Plot solution if desired
        if with_solution:
            # Solve maze if it hasn't been solved already
            if self.solution is None:
                self.solve()
            
            # Initialize path image to be all white
            path_map = np.ones((self.n_rows*self.n_cols, 3))
            # Color in the cells along the solution path
            for index, p in enumerate(self.solution):
                path_map[p] = solution_rgb
                
            # Plot the solution path
            plt.imshow(path_map.reshape([self.n_rows, self.n_cols, 3]), 
                       alpha=solution_alpha)
            
    def solve(self):
        """
        Computes the shortest path through the maze.
        """
        # Compute the brute force solution
        self._solve_brute()
        
        # Remove detours by making sure each cell
        # is only visited once
        i = 0
        while i < len(self.solution):
            # Find last time a given cell is visited
            last = i
            for j in range(i + 1, len(self.solution)):
                if self.solution[j] == self.solution[i]:
                    last = j
            # Cut out detour
            self.solution = self.solution[:i] + self.solution[last:]
            i += 1
    
    def _solve_brute(self):
        """
        Computes a path through the maze by brute force by
        consistently trying to take a right turn if possible.
        """
        # Begin in top-left corner
        cell = self.start_index
        direction = 'E'
        path = [cell]

        n_rows = self.n_rows
        n_cols = self.n_cols
        
        # Set preferred order of directions to always turn right
        travel_bias = {'E': ['S', 'E', 'N', 'W'], 
                       'S': ['W', 'S', 'E', 'N'], 
                       'W': ['N', 'W', 'S', 'E'], 
                       'N': ['E', 'N', 'W', 'S']}

        # Set move limit: no cell should be visited more than 4 times
        max_moves = 4*n_rows*n_cols
        
        # Initialize counter
        moves = 0
        
        # Explore maze until end is reached or move limit is reached
        while cell != self.end_index and moves < max_moves:
            
            # Cycle through moves in order of preference until one 
            # works
            for new_direction in travel_bias[direction]:
                if self.is_legal_move(cell, new_direction):
                    direction = new_direction
                    cell = next_cell(cell, new_direction, n_cols)
                    break

            moves += 1
            path.append(cell)
            
        self.solution = path
            
    def is_legal_move(self, cell, direction):
        """
        Returns whether a desired move is allowed (stays within the 
        maze and doesn't go through a wall)
        
        Args:
            cell (int): location from which to make the move
            direction (str): desired movement direction, one of 'E', 
                'S', 'W', or 'N'
            
        Returns:
            is_legal (bool): whether or not 
        """
        trial_cell = next_cell(cell, direction, self.n_cols) 
        potential_wall = (min(cell, trial_cell), max(cell, trial_cell))
                    
        is_in_bounds = in_bounds(cell, direction, self.n_rows, self.n_cols)
        no_wall = (potential_wall not in self.maze_walls)
        
        is_legal = (is_in_bounds and no_wall)
        return is_legal


def find_component(array, index):
    """
    Find method for union-find/disjoint set data structure. Returns
    the connected component to which the cell at the given index
    belongs.
    
    Args:
        array (array-like)
        index (int): the index of the cell whose connected component
            we want to find
    
    Returns:
        index (int): index of the cell representing the connected 
            component
    """
    while array[index] != -1:
        index = array[index]
    return index


def rotate_line(points):
    """
    Takes lists of the x-coordinates and y-coordinates of the centers
    of two maze cells and returns lists of the x-coordinates and
    y-coordinates of the wall separating the two cells.

    Args:
        points ((float, float), (float, float)): a tuple of
            x-coordinates of maze cell centers followed by tuple of
            y-coordinates of the same cells

    Returns:
        x_vec2, y_vec2 ((float, float), (float, float)): tuple of
            x-coordinates of the endpoints of the line forming the
            maze wall between the two cells, followed by the
            y-coordinates of the wall endpoints
    """
    x_vec = points[0]
    y_vec = points[1]
    if y_vec[0] == y_vec[1]:
        x_vec2 = (np.mean(x_vec), np.mean(x_vec))
        y_vec2 = (y_vec[0] - 0.5, y_vec[0] + 0.5)
    else:
        x_vec2 = (x_vec[0] - 0.5, x_vec[0] + 0.5)
        y_vec2 = (np.mean(y_vec), np.mean(y_vec))
    return x_vec2, y_vec2


def in_bounds(cell, direction, n_rows, n_cols):
    """
    Determine whether or not it's possible to keep moving in a given
    direction without hitting the edge of the maze

    Args:
        cell (int): index of cell from which the move is to be taken
        direction (str): direction of the move; one of 'E', 'S', 'W',
            or 'N'
        n_rows (int): the number of rows in the maze
        n_cols (int): the number of columns in the maze

    Returns:
        (bool): whether or not the move is within the boundaries of
            the maze
    """
    if direction == 'E':
        return (cell + 1) % n_cols != 0
    elif direction == 'S':
        return (cell // n_cols) + 1 < n_rows
    elif direction == 'W':
        return (cell % n_cols) > 0
    else:  # direction == 'N'
        return (cell // n_cols) > 0


def next_cell(cell, direction, n_cols):
    """
    Return the next cell, given the direction of travel
    """
    if direction == 'E':
        return cell + 1
    elif direction == 'S':
        return cell + n_cols
    elif direction == 'W':
        return cell - 1
    else:  # direction == 'N'
        return cell - n_cols

# Luke Weber, 11398889
# CptS 570, HW #5
# Created 11/25/2016

# Problem 3

"""
SUMMARY

Q-Learning (reinforcement learning) solution for 10x10 grid-based world,
using both epsilon-greedy and Boltzmann explore/exploit policies

------------------------------------------------------------------------

You are given a Gridworld environment that is defined as follows:

State space: GridWorld has 10x10 = 100 distinct states. The start state is the top left cell.
The gray cells are walls and cannot be moved to.

Actions: The agent can choose from up to 4 actions (left, right, up, down) to move around.

Environment Dynamics: GridWorld is deterministic, leading to the same new state given
each state and action

Rewards: The agent receives +1 reward when it is in the center square (the one that shows
R 1.0), and -1 reward in a few states (R -1.0 is shown for these). The state with +1.0 reward
is the goal state and resets the agent back to start.

In other words, this is a deterministic, finite Markov Decision Process (MDP). Assume the
discount factor β=0.9.

Implement the Q-learning algorithm (slide 46) to learn the Q values for each state-action pair.
Assume a small fixed learning rate α=0.01.

Experiment with different explore/exploit policies:
1) -greedy. Try  values 0.1, 0.2, and 0.3.
2) Boltzman exploration. Start with a large temperature value T and follow a fixed scheduling
rate.
"""

import math
import time
import copy
import random
import operator
import numpy as np

class GridWorld:
    """
    Grid-based world with only lateral and horizontal moves possible;
    just used as a static reference for our policy
    """

    # World & properties
    grid = None
    num_row = None
    num_col = None

    # Constants
    wall_rep = "_"
    pos_rep = "*"
    goal_rep = "+"
    bad_rep = "-"

    def get_cell(self, pos):
        return self.grid[pos[0]][pos[1]]
    
    def __str__(self):
        """
        Print world, with no additional information
        """

        # Get largest width cell for formatting
        max_width = 0
        for row in self.grid:
            for cell in row:
                if len(cell) > max_width:
                    max_width = len(cell)

        # Construct string rep
        title = "WORLD\n"
        this_str = ""
        sep = "=" * ((len(self.grid) + 4) * max_width) + "\n"
        for i, row in enumerate(self.grid):
            for j, cell in enumerate(row):
                pad = (max_width - len(cell) + 1) * " "
                this_str += cell
                this_str += pad
            this_str += "\n"
        this_str = title + sep + this_str + sep
            
        return this_str

    def __init__(self, grid):
        """
        Constructor: storing all initial information that will define
        all properties of the world
        """

        self.grid = grid
        self.num_row = len(grid)
        self.num_col = len(grid[0])

class Policy:
    """
    Policy -- action chooser -- for standard Markov Decision Process (MDP)
    using Q-values directly;
    REF: https://en.wikipedia.org/wiki/Q-learning
    """

    world = None
    actions = None
    pos = None # [row, col]
    start_pos = None
    goal_pos = None

    discount_factor = None
    learning_rate = None
    q_matrix = None
    rew_matrix = None
    
    def reset(self):
        """
        Reset policy back to goal state; typically done after we've reached
        goal state in one expedition through world;
        NOTE: Still keep all knowledge about world
        """

        self.pos = copy.deepcopy(self.start_pos)

    def next(self):
        """
        Pick next action based on both current position and policy decision;
        NOTE: Implemented by inheritors
        """
        
        raise NotImplementedError

    def action_to_pos(self, action):
        """
        Deterministically convert an action to a new state/position,
        keeping this within boundaries of world
        """

        pos = copy.deepcopy(self.pos)
        if action == self.actions[0]:
            # Up
            if pos[0] != 0: pos[0] -= 1
        elif action == self.actions[1]:
            # Right
            if pos[1] != (self.world.num_col - 1): pos[1] += 1
        elif action == self.actions[2]:
            # Down
            if pos[0] != (self.world.num_row - 1): pos[0] += 1
        elif action == self.actions[3]:
            # Left
            if pos[1] != 0: pos[1] -= 1
        else:
            # Invalid
            raise RuntimeError

        # If we've hit a wall, revert back to original pos
        if self.world.get_cell(pos) == GridWorld.wall_rep:
            pos = self.pos

        return pos

    def get_best_action(self, pos):
        """
        Get best action (based on Q-values) from given position such that
        actions of equal likelihood are drawn randomly
        """

        # Get moves available
        pos_index = pos[0] * self.world.num_col + pos[1]
        moves = self.q_matrix[pos_index]
        
        # Shuffle up moves, so we move randomly for actions
        # with equal Q-values
        action_dict = {}
        for move_i, move_q in enumerate(moves):
            action_dict[move_i] = move_q
        action_items = list(action_dict.items())
        random.shuffle(action_items)

        # Get best move val and index (i.e. highest Q)
        highest_move = -1
        highest_move_index = -1
        for move_i, move_q in action_items:
            if move_q > highest_move:
                highest_move = move_q
                highest_move_index = move_i

        return highest_move_index, highest_move

    def get_Q_matrix_pos(self, pos):
        """
        Get Q-matrix position from given position
        """

        return pos[0] * self.world.num_col + pos[1]

    def get_reward(self, pos):
        """
        Parse reward, store it, and return it
        """
        
        c = self.world.grid[pos[0]][pos[1]]
        self.rew_matrix[pos[0]][pos[1]] = float(c)

        return float(c)

    def __str__(self):
        """
        Print world with current agent position and the direction at
        each cell which represents the maximum Q-value
        """

        action_sym = ['^', '>', 'v', '<']

        title = "POLICY\n"
        pad_len = 3
        this_str = ""
        sep = "=" * ((len(self.world.grid) + pad_len + 1) * 2) + "\n"
    
        for i, row in enumerate(self.world.grid):
            for j, cell in enumerate(row):

                new_elem = ""

                # Add new element contents
                if self.pos == [i, j]:
                    # Current position
                    new_elem = "*"
                if cell == GridWorld.wall_rep:
                    # Don't show walls
                    new_elem = " "
                elif [i, j] == self.goal_pos:
                    # Show goal
                    new_elem += GridWorld.goal_rep
                elif self.world.get_cell([i, j]) == "-1":
                    # Show bad spots
                    new_elem += GridWorld.bad_rep
                else:
                    # Print direction of best action, but only if
                    # our best Q-value is non-zero, meaning we
                    # are not just showing a random direction, but
                    # something which has been discovered
                    a_i, a_q = self.get_best_action([i, j])
                    new_elem += action_sym[a_i]

                # Pad
                new_elem += " " * (pad_len - len(new_elem))
                this_str += new_elem
            this_str += "\n"
        this_str = title + sep + this_str + sep
            
        return this_str

    def __init__(self, world, start_pos, goal_pos, discount_factor,
                 learning_rate):
        """
        Constructor: pass in reference to world and some Q-learning
        based parameters; using explicit |S|x|A| table to represent Q
        """

        actions = ['u', 'r', 'd', 'l']
        num_states = world.num_row * world.num_col
        num_actions = len(actions)

        self.world = world
        self.start_pos = copy.deepcopy(start_pos)
        self.goal_pos = copy.deepcopy(goal_pos)
        self.pos = copy.deepcopy(start_pos)
        self.actions = actions
        self.q_matrix = np.zeros((num_states, num_actions))
        self.rew_matrix = np.zeros((world.num_row, world.num_col))
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

class EpsilonGreedyPolicy(Policy):
    """
    Take random action if randomly drawn number between 0 and 1 is less
    than epsilon, otherwise we pick best option as determined by Q-vals
    """
    
    epsilon = None

    def next(self):
        """
        TODO
        """

        orig_state = copy.deepcopy(self.pos)    # state s
        pos_index = self.get_Q_matrix_pos(self.pos)
        moves = self.q_matrix[pos_index]

        # Find best action
        highest_move_index, highest_move_q = self.get_best_action(
            orig_state)

        # Roll die: explore or exploit?
        rand_val = random.uniform(0, 1)
        if rand_val < self.epsilon: picked_move = random.randint(0, 3)
        else: picked_move = highest_move_index

        # Carry out action
        action = self.actions[picked_move]      # action a
        
        self.pos = self.action_to_pos(action)   # state s'

        # Temporal Difference (TD) update
        reward = self.get_reward(self.pos)              # reward r = R(s')
        # Estimate of optimal future value: max{a'}{Q(s', a')}
        opt_future_i, opt_future_q = self.get_best_action(self.pos)
        # Q-value of moving to state s' from s via action a
        this_q = self.q_matrix[pos_index][picked_move]  # Q(s, a)
        x = self.learning_rate * (reward + self.discount_factor * \
                                  opt_future_q - this_q)
        self.q_matrix[pos_index][picked_move] += x

        # Return true if we've reached goal state
        if self.pos == self.goal_pos: return False
        else: return True
    
    def __init__(self, world, start_pos, goal_pos, discount_factor,
                 learning_rate, epsilon):
        Policy.__init__(self, world, start_pos, goal_pos, discount_factor,
                        learning_rate)
        self.epsilon = epsilon

class BoltzmannExplorationPolicy(Policy):
    """
    TODO
    REF: https://www.cs.cmu.edu/afs/cs/academic/class/15381-s07/
    www/slides/050107reinforcementLearning1.pdf
    """

    temperature = None

    def reset(self):
        """
        Overwriting reset function in base policy, but then calling their
        implementation after we've decremented our temperature; don't
        let temperature get less than 1
        """
               
        if self.temperature >= 1: self.temperature -= 1
        Policy.reset(self)

    def next(self):
        """
        TODO
        """

        orig_state = copy.deepcopy(self.pos)    # state s
        pos_index = self.get_Q_matrix_pos(self.pos)
        moves = self.q_matrix[pos_index]

        # Circumvent math issues with temperature actually being 0
        if self.temperature > 0:
            # Compute action probabilities using temperature; when
            # temperature is high, we're treating values of very different
            # Q-values as more equally choosable
            action_probs_numes = []
            denom = 0
            for m in moves:
                val = math.exp(m / self.temperature)
                action_probs_numes.append(val)
                denom += val
            action_probs = [x / denom for x in action_probs_numes]

            # Pick random move, in which moves with higher probability are
            # more likely to be chosen, but it is obviously not guaranteed
            rand_val = random.uniform(0, 1)
            prob_sum = 0
            for i, prob in enumerate(action_probs):
                prob_sum += prob
                if rand_val <= prob_sum:
                    picked_move = i
                    break
        else:
            # Here, we're totally cold; meaning, we're just exploiting
            picked_move, picked_move_q = self.get_best_action(orig_state)

        # Carry out action
        action = self.actions[picked_move]      # action a
        self.pos = self.action_to_pos(action)   # state s'

        # Temporal Difference (TD) update
        reward = self.get_reward(self.pos)              # reward r = R(s')
        # Estimate of optimal future value: max{a'}{Q(s', a')}
        opt_future_i, opt_future_q = self.get_best_action(self.pos)
        # Q-value of moving to state s' from s via action a
        this_q = self.q_matrix[pos_index][picked_move]  # Q(s, a)
        x = self.learning_rate * (reward + self.discount_factor * \
                                  opt_future_q - this_q)
        self.q_matrix[pos_index][picked_move] += x

        # Return true if we've reached goal state
        if self.pos == self.goal_pos: return False
        else: return True

    def __str__(self):
        orig_state = copy.deepcopy(self.pos)
        pos_index = self.get_Q_matrix_pos(self.pos)
        moves = self.q_matrix[pos_index]
        action_probs_numes = []
        denom = 0
        for m in moves:
            val = math.exp(m / self.temperature)
            action_probs_numes.append(val)
            denom += val
        action_probs = [x / denom for x in action_probs_numes]
        
        this_str = Policy.__str__(self)
        this_str += "temperature: " + str(self.temperature) + "\n"
        this_str += "position probabilities: " + str(action_probs)
        return this_str
    
    def __init__(self, world, start_pos, goal_pos, discount_factor,
                 learning_rate, temperature):
        Policy.__init__(self, world, start_pos, goal_pos, discount_factor,
                        learning_rate)
        self.temperature = temperature

def create_epsilon_greedy_policy(world, epsilon):
    """
    Create epsilon-greedy policy with given epsilon value
    """

    # Generic policy params
    discount_factor = 0.9
    learning_rate = 0.01
    start_pos = [0, 0]  # top-left
    goal_pos = [5, 5]   # middle

    return EpsilonGreedyPolicy(world, start_pos, goal_pos,
                               discount_factor, learning_rate, epsilon)

def create_boltzmann_policy(world, temperature):
    """
    Create Boltzmann exploration policy with given starting temperature
    """

    # Generic policy params
    discount_factor = 0.9
    learning_rate = 0.01
    start_pos = [0, 0]  # top-left
    goal_pos = [5, 5]   # middle

    return BoltzmannExplorationPolicy(world, start_pos, goal_pos,
                               discount_factor, learning_rate, temperature)

def create_world():
    """
    Create grid specific to this assignment: +1 reward in center, some
    -1 rewards around, some walls, and rest are 0 reward
    """

    # World
    num_cols = 10
    num_rows = 10
    grid = [['0' for col in range(num_cols)]
             for row in range(num_rows)]

    # Walls
    for i in range(1, 5): grid[2][i] = GridWorld.wall_rep
    for i in range(6, 9): grid[2][i] = GridWorld.wall_rep
    for i in range(3, 8): grid[i][4] = GridWorld.wall_rep
    
    # Goal
    grid[5][5] = '1'

    # Losses
    grid[4][5] = '-1'
    grid[4][6] = '-1'
    grid[5][6] = '-1'
    grid[5][8] = '-1'
    grid[6][8] = '-1'
    grid[7][3] = '-1'
    grid[7][5] = '-1'
    grid[7][6] = '-1'
    
    return GridWorld(grid)

def print_menu(null):
    print("| 'q N' to see Q-matrix for N-th policy,",
          "\n| 'p N' to see resulting policy map for N-th policy,",
          "\n| 'r N' to run greedily the N-th policy through world,",
          "\n| 'help me' to show menu, or",
          "\n| 'exit now' to stop program")
    return True

def run_policy(policy):
    # TODO: check if policy uses epsilon
    print("Greedily running policy through world:")
    print()

    # Make sure we only exploit
    if isinstance(policy, BoltzmannExplorationPolicy):
        policy.temperature = 0 #0.0001
    else: policy.epsilon = 0
    
    time.sleep(0.5)
    while policy.next() == True:
        print(policy)
        time.sleep(0.5)
    print(policy)
    return True

def print_policy(policy):
    print("Printing policy:")
    print()
    print(policy)
    return True

def print_policy_q_matrix(policy):
    # TODO: Possibly print our own q matrix here
    print("Printing Q-values (order of actions is",
          "[Up, Right, Down, Left]):")
    print()

    this_str = ""
    for row in policy.q_matrix:
        for i, col in enumerate(row):
            this_str += str(col)
            if i != (len(row) - 1): this_str += ","
        this_str += "\n"

    # Write to file
    with open("data.csv", "w") as file:
        file.write(this_str)

    #print(this_str)
    
    return True

def exit_program(null):
    print("Exiting program...")
    return False

def run_stats_menu(policies):
    """
    Loop menu for user to get information about the given policies.
    """
    
    print("Menu: print statistics on the", len(policies),
          "given policies:")
    print_menu(None)

    commands = {"p": print_policy,
                "q": print_policy_q_matrix,
                "r": run_policy,
                "help": print_menu,
                "exit": exit_program}

    while True:
        print()
        cmd = input("Command: ")
        tokens = cmd.split()

        # Check valid command length
        # NOTE: Checking for menu is hard-coded (not great)
        if (len(tokens) > 2 or len(tokens) < 2):
            print("Error: invalid number of arguments")
            continue

        # Check valid command
        if tokens[0] not in commands:
            print("Error: invalid command")
            continue

        # Check valid policy number
        if tokens[0] != "help" and tokens[0] != "exit":
            if not rep_int(tokens[1]):
                print("Error: argument type not int")
                continue
            if int(tokens[1]) > len(policies) or int(tokens[1]) <= 0:
                print("Error: argument not in range")
                continue

            # Perform command
            # NOTE: exit() doesn't care if you pass an object into it
            policy = policies[int(tokens[1]) - 1]
            if not commands[tokens[0]](policy): return
        else:
            if not commands[tokens[0]](None): return

def rep_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def matrix_diff(matrix_one, matrix_two):
    total_diff = 0
    for i, row in enumerate(matrix_one):
        # Check if max move is different
        index_one, value_one = max(enumerate(matrix_one[i]),
                                   key=operator.itemgetter(1))
        index_two, value_two = max(enumerate(matrix_two[i]),
                                   key=operator.itemgetter(1))
        if index_one != index_two: total_diff += 1
        #for j, col in enumerate(row):
            #total_diff += abs(matrix_one[i][j] - matrix_two[i][j])
    
    return total_diff

def main():
    """
    Facilitates world create, policy creation with that world, and...
    """

    print("Welcome to my Q-learning implementation!")
    print("Developed by Luke Weber on 11/25/2016")

    tab = " "
    world = create_world()
    policies = []
    convergence_thresh = 0  # was 0.001
    converge_count = 1000 #100    # number of times must hit threshold in a row
    est_train_time = "5-7 mins"
        
    # Policy-generating functions and arrays of params

    policy_funcs = [create_epsilon_greedy_policy,
                    create_boltzmann_policy]
    policy_params = [("epsilon", [0.1, 0.2, 0.3]),
                     ("temperature", [2000, 1000, 100, 10, 1])]
    
    # Show world
    print()
    print(world)

    # Show policies
    print("Running two different explore/exploit policies:")
    print(tab, "A. Epsilon-greedy,", "epsilon =",
          policy_params[0][1])
    print(tab, "B. Boltzmann exploration,", "temperature =",
          policy_params[1][1])
    print()

    # Run each policy until its convergence with
    # its different list of arguments to try
    total_time = 0
    print("Doing Q-learning (" + est_train_time + "):")
    for i, policy_func in enumerate(policy_funcs):
        param_name, param_list = policy_params[i]
        for param in param_list:
            policy = policy_func(world, param)
            print(tab, str(len(policies) + 1) + ".", "Running",
                  policy.__class__.__name__ + ",", param_name,
                  "=", param)
            
            ts = time.clock()
            num_iter = 0
            this_conv_count = converge_count
            
            while True:
                num_iter += 1
                last_q_matrix = copy.deepcopy(policy.q_matrix)

                # Run policy until reaches goal
                while policy.next() == True: pass
                policy.reset()

                # Check convergence in back-to-back epochs
                if matrix_diff(last_q_matrix, policy.q_matrix) \
                   <= convergence_thresh:
                    this_conv_count -= 1
                    if this_conv_count == 0: break
                # Reset
                else: this_conv_count = converge_count
            
            te = time.clock()
            total_time += te - ts
            print(tab, tab, ("...Converged in {0:.2f} seconds and {1} " +
                  "iterations").format((te - ts), (num_iter - converge_count)))

            # Store for later reference
            policies.append(policy)
            
    print(("...Completed training in {0:.2f} seconds with convergence " +
           "threshold of " + str(convergence_thresh))
          .format(total_time))
    print()

    # Run menu
    run_stats_menu(policies)

# Get ball rolling
main()

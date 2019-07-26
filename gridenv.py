import numpy as np
import gym
import gym.spaces
from gym.utils import seeding
from gym.envs.registration import register
import matplotlib.pyplot as plt
import imageio
import pickle as pkl
from envShape import allRooms

z_representation = 120

def stripWall(grid):
    grid_new = np.zeros((grid.shape[0] -2, grid.shape[1]- 2 ))
    for i in range(len(grid)-1):
        for j in range(len(grid[0])-1):
            if i>0 and j>0:
                grid_new[i-1][j-1] = grid[i][j][0]
    return grid_new

def get_middle_rep1(wts, env1_state):
    # with open(wts_file1, 'rb') as handle:
    #     wts = pkl.load(handle)
    sz = env1_state.shape[0]
    state_flatten = env1_state.reshape([-1])

    # output = np.matmul(wts["W"].T,state_flatten) + wts["b"]
    output = np.matmul(wts['W'].T, state_flatten) + wts["b"]
    output = np.maximum(output,0)

    return output

def get_middle_rep2(wts, env2_state):
    # with open(wts_file2, 'rb') as handle:
    #     wts = pkl.load(handle)
    sz = env2_state.shape[0]
    state_flatten = env2_state.reshape([-1])

    output = np.matmul(wts['W'].T, state_flatten) + wts["b"]
    output = np.maximum(output,0)

    return output



class GridRooms(gym.Env):
    """
    Navigational tasks where reaching goal yields a reward
    """
    def __init__(self, roomName='env_train',
                 randomStart=True, randomGoal=True,
                 dense=False):

        if roomName not in allRooms:
            raise TypeError("Invalid room name \'%s\'" % roomName)

        self.wts = None
        with open("wts_noisy_file1", 'rb') as handle:
            self.wts = pkl.load(handle)

        self.randomStart = randomStart
        self.randomGoal = randomGoal
        self.dense = dense

        self.room = allRooms[roomName]
        self.height = len(self.room)
        self.width = len(self.room[0])

        # Set state and action spaces
        self.action_space = gym.spaces.Discrete(4)
        # self.observation_space = gym.spaces.Box(
        #     low=0.0, high=1.0, shape=(self.height, self.width, 1), dtype='uint8')
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape= (z_representation,), dtype='float64')

        #  Validate room
        chrs = set()
        for i in range(self.height):
            chrs = chrs.union(set(self.room[i]))

        if len(chrs.difference(set([' ', 'G', 'S', '#']))) != 0:
            raise TypeError("Invalid values in the grid")

        #  Get start states
        self.startStates = []
        self.goalStates = []
        for i in range(self.height):
            for j in range(self.width):
                if randomStart is True:
                    if self.room[i][j] in ['S', ' ', 'G']:
                        self.startStates.append((i, j))
                elif self.room[i][j] == 'S':
                    self.startStates.append((i, j))

                if randomGoal is True:
                    if self.room[i][j] in ['S', ' ', 'G']:
                        self.goalStates.append((i, j))
                elif self.room[i][j] == 'G':
                    self.goalStates.append((i, j))

        if len(self.goalStates) == 0 or len(self.startStates) == 0:
            raise ValueError("Either goal or start states not specified")

        goalInd = np.random.randint(len(self.goalStates))
        self.goal = self.goalStates[goalInd]

        if dense:
            self.__createLevels(self.goal)

        self.done = False
        self.steps = 0
        self.img = None
        self.images = []
        self.__actionSpace()

        if len(self.startStates) == 0:
            self.startStates.append((1, 1))

        self.reset()

    def __createLevels(self, goal):
        stack = [goal]
        level = {goal: 0}

        while len(stack) > 0:
            cur = stack.pop()
            x = cur[0]
            y = cur[1]
            vals = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dx, dy in vals:
                if self.room[x + dx][y + dy] != '#':
                    pos = (x + dx, y + dy)
                    if pos not in level:
                        level[pos] = level[cur] - 1
                        stack.insert(0, pos)
        self.level = level

    def __resetStart(self):
        #  goalInd = np.random.randint(len(self.goalStates))
        #  self.goal = self.goalStates[goalInd]
        while True:
            startInd = np.random.randint(len(self.startStates))
            self.start = self.startStates[startInd]
            if (self.start != self.goal):
                break

    def __actionSpace(self):
        self.LEFT = 0
        self.RIGHT = 1
        self.UP = 2
        self.DOWN = 3

    def __getStateRep(self):
        grid = np.zeros((self.height, self.width, 1))
        grid[self.pos[0], self.pos[1], 0] = 1
        stripped_state = stripWall(grid)
        return get_middle_rep1(self.wts, stripped_state)



    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def retPos(self):
        return self.pos

    def step(self, action):
        self.steps += 1
        x = self.pos[0]
        y = self.pos[1]
        oldPos = tuple(self.pos)
        print("check2 ", action )
        if action == self.LEFT:
            if self.room[x - 1][y] != '#':
                self.pos = (x - 1, y)
        elif action == self.RIGHT:
            if self.room[x + 1][y] != '#':
                self.pos = (x + 1, y)
        elif action == self.UP:
            if self.room[x][y - 1] != '#':
                self.pos = (x, y - 1)
        elif action == self.DOWN:
            if self.room[x][y + 1] != '#':
                self.pos = (x, y + 1)

        rew = 0.0
        if self.dense:
            rew = (self.level[self.pos] - self.level[oldPos]) * 0.2

        if tuple(self.pos) == tuple(self.goal):
            rew = 10.0
            self.done = True

        if self.steps >= 500:
            #  rew = 10
            self.done = True

        return self.__getStateRep(), rew, self.done, {}

    def reset(self):
        self.__resetStart()
        self.pos = self.start
        self.steps = 0
        self.img = None
        self.done = False
        return self.__getStateRep()

    def render(self, disp=1):
        grid = np.zeros((self.height, self.width))
        for i in range(self.height):
            for j in range(self.width):
                if self.room[i][j] != '#':
                    grid[i, j] = 0
                else:
                    grid[i, j] = 1
        grid = 255 - grid * 220
        grid = np.transpose(np.asarray([grid, grid, grid]), [1, 2, 0])
        grid = grid.astype(np.uint8)
        grid[self.start[0], self.start[1]] = (0, 0, 255)
        grid[self.goal[0], self.goal[1]] = (0, 255, 0)
        grid[self.pos[0], self.pos[1]] = (255, 0, 0)
        if self.img is None:
            plt.axis('off')
            self.img = plt.imshow(grid)
        else:
            self.img.set_data(grid)

        self.images.append(grid)
        if disp != 0:
            plt.pause(0.05)
            plt.draw()
        return grid

    def close(self, savePath=None):
        if savePath:
            imageio.mimsave(savePath, self.images)
        self.images = []
        plt.close()


class GridRooms2(gym.Env):
    """
    Navigational tasks where reaching goal yields a reward
    """
    def __init__(self, roomName='env_train_noisy',
                 randomStart=True, randomGoal=True,
                 dense=False):

        if roomName not in allRooms:
            raise TypeError("Invalid room name \'%s\'" % roomName)


        self.wts = None
        with open("wts_noisy_file2", 'rb') as handle:
            self.wts = pkl.load(handle)

        self.randomStart = randomStart
        self.randomGoal = randomGoal
        self.dense = dense

        self.room = allRooms[roomName]
        self.height = len(self.room)
        self.width = len(self.room[0])

        # Set state and action spaces
        self.action_space = gym.spaces.Discrete(4)
        #self.observation_space = gym.spaces.Box(
        #    low=0.0, high=1.0, shape=(self.height, self.width, 1), dtype='uint8')
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(z_representation,), dtype='float64')

        #  Validate room
        chrs = set()
        for i in range(self.height):
            chrs = chrs.union(set(self.room[i]))

        if len(chrs.difference(set([' ', 'G', 'S', '#']))) != 0:
            raise TypeError("Invalid values in the grid")

        #  Get start states
        self.startStates = []
        self.goalStates = []
        for i in range(self.height):
            for j in range(self.width):
                if randomStart is True:
                    if self.room[i][j] in ['S', ' ', 'G']:
                        self.startStates.append((i, j))
                elif self.room[i][j] == 'S':
                    self.startStates.append((i, j))

                if randomGoal is True:
                    if self.room[i][j] in ['S', ' ', 'G']:
                        self.goalStates.append((i, j))
                elif self.room[i][j] == 'G':
                    self.goalStates.append((i, j))

        if len(self.goalStates) == 0 or len(self.startStates) == 0:
            raise ValueError("Either goal or start states not specified")

        goalInd = np.random.randint(len(self.goalStates))
        self.goal = self.goalStates[goalInd]

        if dense:
            self.__createLevels(self.goal)

        self.done = False
        self.steps = 0
        self.img = None
        self.images = []
        self.__actionSpace()

        if len(self.startStates) == 0:
            self.startStates.append((1, 1))

        self.reset()

    def __createLevels(self, goal):
        stack = [goal]
        level = {goal: 0}

        while len(stack) > 0:
            cur = stack.pop()
            x = cur[0]
            y = cur[1]
            vals = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dx, dy in vals:
                if self.room[x + dx][y + dy] != '#':
                    pos = (x + dx, y + dy)
                    if pos not in level:
                        level[pos] = level[cur] - 1
                        stack.insert(0, pos)
        self.level = level

    def __resetStart(self):
        #  goalInd = np.random.randint(len(self.goalStates))
        #  self.goal = self.goalStates[goalInd]
        while True:
            startInd = np.random.randint(len(self.startStates))
            self.start = self.startStates[startInd]
            if (self.start != self.goal):
                break

    def __actionSpace(self):
        self.LEFT = 0
        self.RIGHT = 1
        self.UP = 2
        self.DOWN = 3

    def __getStateRep(self):
        grid = np.zeros((self.height, self.width, 1))
        grid[self.pos[0], self.pos[1], 0] = 1
        stripped_state = stripWall(grid)
        return get_middle_rep2(self.wts, stripped_state)


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def retPos(self):
        return self.pos

    def step(self, action):
        self.steps += 1
        x = self.pos[0]
        y = self.pos[1]
        oldPos = tuple(self.pos)
        print("check1 ", action )
        if action == self.RIGHT:
            print("left outside if")

            if self.room[x + 1][y] != '#':
                self.pos = (x + 1, y)
        elif action == self.LEFT:
            print("right outside if")

            if self.room[x - 1][y] != '#':
                self.pos = (x - 1, y)
        elif action == self.DOWN:
            print("down outside if")
            if self.room[x][y + 1] != '#':
                print("down inside if")
                self.pos = (x, y + 1)
        elif action == self.UP:
            print("up outside if")

            if self.room[x][y - 1] != '#':
                self.pos = (x, y - 1)

        rew = 0.0
        if self.dense:
            rew = (self.level[self.pos] - self.level[oldPos]) * 0.2

        if tuple(self.pos) == tuple(self.goal):
            rew = 10.0
            self.done = True

        if self.steps >= 500:
            #  rew = 10
            self.done = True

        return self.__getStateRep(), rew, self.done, {}

    def reset(self):
        self.__resetStart()
        self.pos = self.start
        self.steps = 0
        self.img = None
        self.done = False
        return self.__getStateRep()

    def render(self, disp=1):
        grid = np.zeros((self.height, self.width))
        for i in range(self.height):
            for j in range(self.width):
                if self.room[i][j] != '#':
                    grid[i, j] = 0
                else:
                    grid[i, j] = 1
        grid = 255 - grid * 220
        grid = np.transpose(np.asarray([grid, grid, grid]), [1, 2, 0])
        grid = grid.astype(np.uint8)
        grid[self.start[0], self.start[1]] = (0, 0, 255)
        grid[self.goal[0], self.goal[1]] = (0, 255, 0)
        grid[self.pos[0], self.pos[1]] = (255, 0, 0)
        if self.img is None:
            plt.axis('off')
            self.img = plt.imshow(grid)
        else:
            self.img.set_data(grid)

        self.images.append(grid)
        if disp != 0:
            plt.pause(0.05)
            plt.draw()
        return grid

    def close(self, savePath=None):
        if savePath:
            imageio.mimsave(savePath, self.images)
        self.images = []
        plt.close()


register(
    id='no_noise-v2',
    entry_point='gridenv:GridRooms',
    kwargs={'roomName': 'env_train', 'dense': True,
            'randomStart': False, 'randomGoal': False}
)

register(
    id='noise-v2',
    entry_point='gridenv:GridRooms2',
    kwargs={'roomName': 'env_train_noisy', 'dense': True,
            'randomStart': False, 'randomGoal': False}
)

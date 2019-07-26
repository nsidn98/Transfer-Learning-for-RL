import numpy as np

class flippedStates():
    def __init__(self,input_shape):
        self.input_shape = input_shape
        self.identity = np.identity(input_shape, dtype = np.float32)

    def getStates(self):
        original_states = []
        flipped_states = []
        for i in range(self.input_shape):
            original_states.append(self.identity[i])
            flipped_states.append(self.identity[-(i+1)])

        return np.array(original_states),np.array(flipped_states)

class shapeStates():
    def __init__(self,input_shape):
        self.input_shape = input_shape
        self.identity = np.identity(input_shape, dtype = np.float32)

    def getStates(self,flip=False):
        original_states = []
        new_states = []
        for i in range(self.input_shape):
            original_states.append(self.identity[i])
            if flip:
                new_array = list(self.identity[-(i+1)])
                new_array.append(0)
                new_array.insert(0,0)
                # new_array = [0, flipped(orig_state), 0]
                new_array = np.array(new_array)
                # print(new_array)
                new_states.append(new_array)
            else:
                new_array = list(self.identity[i])
                new_array.append(0)
                new_array.insert(0,0)
                # print(new_array)
                new_array = np.array(new_array)
                new_states.append(new_array)
                # new_array = [0, orig_state, 0]

        return np.array(original_states), np.array(new_states)

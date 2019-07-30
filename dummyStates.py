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
    def __init__(self,input_shape_1,input_shape_2):
        self.input_shape_1 = input_shape_1
        self.input_shape_2 = input_shape_2
        self.identity = np.identity(input_shape_1, dtype = np.float32)

    def getStates(self,flip=False,random=False):
        original_states = []
        new_states = []
        for i in range(self.input_shape_1):
            original_states.append(self.identity[i])
            if flip:
                new_array = list(self.identity[-(i+1)])
                diff = self.input_shape_2 - self.input_shape_1
                for j in range(int(diff/2)):
                    if random:
                        new_array.append(np.random.rand(1)[0])
                        new_array.insert(0,np.random.rand(1)[0])
                    else:
                        new_array.append(0)
                        new_array.insert(0,0)
                new_array = np.array(new_array)
                new_states.append(new_array)
            else:
                new_array = list(self.identity[i])
                diff = self.input_shape_2 - self.input_shape_1
                for j in range(int(diff/2)):
                    if random:
                        new_array.append(np.random.rand(1)[0])
                        new_array.insert(0,np.random.rand(1)[0])
                    else:
                        new_array.append(0)
                        new_array.insert(0,0)
                new_array = np.array(new_array)
                new_states.append(new_array)

        return np.array(original_states), np.array(new_states)

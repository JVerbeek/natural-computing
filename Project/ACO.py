import numpy as np
from mdrnn import LSTMCell
from train_mdrnn import train

'''
Questions/problems:
    Do you not only need m1 to determine the needed connections? It doesn't
    matter if an input path is 0, you'll figure that out automatically if 
    none of the m1 row entries are active; idem for output path.
'''

N_ANTS = 5
N_INPUTS = 16
MAX_PHEROMONE = 10
DEG_FREQ = 2

class Paths():
    def __init__(self, n_inputs):
        self.input = np.zeros(n_inputs, dtype=bool)
        self.m1 = np.zeros((n_inputs, n_inputs), dtype=bool)
        self.m2 = np.zeros(n_inputs, dtype=bool)

class Pheromones():
    def __init__(self, n_inputs):
        self.input = np.ones(n_inputs)
        self.m1 = np.ones((n_inputs, n_inputs))
        self.m2 = np.ones(n_inputs)
    

def generate_paths(pheromones):
    paths = Paths(N_INPUTS)
    
    for ant in range(1, N_ANTS):
        pheromone_sum = np.sum(pheromones.input)
        r = np.random.uniform(0, pheromone_sum - 1)
        input_path = 0
        while r > 0:
            if r < pheromones.input[input_path]:
                paths.input[input_path] = 1
                break
            else:
                r = r - pheromones.input[input_path]
                input_path += 1
        pheromone_sum = np.sum(pheromones.m1[input_path])
        r = np.random.uniform(0, pheromone_sum - 1)
        hidden_path = 0
        while r > 0:
            if r < pheromones.m1[input_path][hidden_path]:
                paths.m1[input_path][hidden_path] = 1
                paths.m2[hidden_path] = 1
                break
            else:
                r = r - pheromones.m1[input_path][hidden_path]
                hidden_path += 1
    return paths
    
def update_pheromones(pheromones, paths, action):
    for i in range(1, len(paths.input)):
        if paths.input[i] == 1:
            if action == 0:  # reward
                pheromones.input[i] = min(pheromones.input[i] \
                                                * 1.15, MAX_PHEROMONE)
            elif action == 1:  # penalize
                pheromones.input[i] *= 0.85
            elif action == 2:  # degrade
                pheromones.input[i] *= 0.9
    for i in range(1, len(paths.m1)):
        for j in range(1, len(paths.m1[i])):
            if paths.m1[i][j] == 1:
                if action == 0:
                    pheromones.m1[i][j] = min(pheromones.m1[i][j] \
                                                * 1.15, MAX_PHEROMONE)
                elif action == 1:
                    pheromones.m1[i][j] *= 0.85
                elif action == 2:
                    pheromones.m1[i][j] *= 0.9
    
                    
if __name__ == "__main__":
    pheromones = Pheromones(N_INPUTS)
    cur_best = 0
    n_epochs = 2
    model = LSTMCell(N_INPUTS, N_INPUTS)
    
    for epoch in range(n_epochs):
        paths = generate_paths(pheromones)
        
        # train model / get fitness
        fitness = train(model, paths)
        
        if fitness > cur_best:
            cur_best = fitness
            update_pheromones(pheromones, paths, 0)
        else:
            update_pheromones(pheromones, paths, 1)
        if epoch % DEG_FREQ == 0:
            update_pheromones(pheromones, paths, 2)
   
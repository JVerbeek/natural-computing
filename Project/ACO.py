import numpy as np
from mdrnn_tf import MDRNNCell
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torch.optim as optim
import math
#from train_mdrnn import train

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
    """
    

    Parameters
    ----------
    pheromones : TYPE
        DESCRIPTION.

    Returns
    -------
    paths : TYPE
        DESCRIPTION.

    """
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
    """
    Update pheromones along specific connections.
    
    Parameters
    ----------
    pheromones : TYPE
        DESCRIPTION.
    paths : Paths object
        DESCRIPTION.
    action : int
        Specifies to reward (0), penalize (1) or degrade (2) the topology 
        specified by the Paths object

    Returns
    -------
    None.

    """
    # This function should be part of the pheromones class probably
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
                    
def train(model, paths):
    """Dummy train function"""
    return -1


class CustomLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(CustomLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, hidden):
        
        hx, cx = hidden
        gates = self.x2h(x) + self.h2h(hx)
    
        gates = gates.squeeze()
        
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)        

        cy = torch.mul(cx, forgetgate) +  torch.mul(ingate, cellgate)        

        hy = torch.mul(outgate, F.tanh(cy))
        
        return (hy, cy)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers=1):
        super(LSTM, self).__init__()
        
        self.input_size=input_size
        self.hidden_size = hidden_size
        self.num_layers=num_layers
        cell_list=[]
        
        
        
        cell_list.append(CustomLSTMCell( self.input_size, self.hidden_size))#the first
        # one has a different number of input channels
        
        for idcell in range(1,self.num_layers):
            cell_list.append(CustomLSTMCell(self.hidden_size, self.hidden_size))
        self.cell_list=nn.ModuleList(cell_list)      
    
    def forward(self, current_input, hidden_state):
        """
        args:
            hidden_state:list of tuples, one for every layer, each tuple should be hidden_layer_i,c_layer_i
            input is the tensor of shape seq_len,Batch,Chans,H,W
        """
        #current_input=input
        next_hidden = []  # hidden states(h and c)
        seq_len = current_input.size(0)

        
        for idlayer in range(self.num_layers):#loop for every layer

            hidden_c = hidden_state[idlayer] # hidden and c are images with several channels
            all_output = []
            output_inner = []            
            for t in range(seq_len):#loop for every step
                hy, cy = self.cell_list[idlayer](current_input,hidden_c)

                output_inner.append(hidden_c)

            next_hidden.append(hidden_c)
            current_input = hidden_c[0]
    
        return next_hidden


def prune_connections(model, paths, output_type="m1"):
    """
    Prune connections in model in accordance with the paths found by the
    algorithm.

    Parameters
    ----------
    model : Model
        Model which is to be pruned
    paths : Paths 
        Paths objects, specifies which paths in the model must be pruned
    output_type : str, optional
        Whether the cell type will be M1 or M2. The default is "m1".

    Returns
    -------
    model: Model
        Pruned model.

    """
    if output_type == "m1":  # Output type M1
        path = torch.from_numpy(paths.m1)
        prune.custom_from_mask(model.x2h, "weight", path)
        prune.custom_from_mask(model.h2h, "weight", path)

    elif output_type == "m2":  # Output type M2
        path = torch.from_numpy(paths.m2)
        prune.custom_from_mask(model.x2h, "weight", path)
        prune.custom_from_mask(model.h2h, "weight", path)
        
    else:
        raise ValueError("output_type kwarg must be m1 or m2")
    
    return model

if __name__ == "__main__":
    
    cell = CustomLSTMCell(1, 1)
    pheromones = Pheromones(N_INPUTS)
    cur_best = 0
    n_epochs = 10
    pop_size = 5
    models = []
    base_model = CustomLSTMCell(N_INPUTS, N_INPUTS)
    
    # Generate population
    for idx in range(pop_size):
        model = base_model
        paths = generate_paths(pheromones)    
        pruned_model = prune_connections(model, paths)
        # Append (model, fitness) tuple
        models.append((pruned_model, 0))

    # Some vars for testing (CHAOS)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Testing data
    train_input = torch.rand(4, 4, 16)
    test_input = torch.rand(4, 4, 16)
    labels = torch.empty(4, 16, dtype=torch.long).random_(4)
    test_labels = torch.empty(4, 16, dtype=torch.long).random_(4)
    
    
    hn = torch.rand(4, 4, 16)
    cn = torch.rand(4, 4, 16)

    for (model, fitness) in models:
        # Train phase for single network
        for epoch in range(n_epochs):

            # forward + backward + optimize
            hn, cn = model(train_input, (hn, cn))
            loss = criterion(hn, labels)
            loss.backward(retain_graph=True)
            optimizer.step()
            
        # Compute fitness on a test set
        ht, ct = model(test_input, (hn, cn))
        fitness = criterion(ht, labels)
    
    population = models
    
    for (model, fitness) in population:
        # Pheromone update round
        if fitness > cur_best:
            cur_best = fitness
            update_pheromones(pheromones, paths, 0)
        else:
            update_pheromones(pheromones, paths, 1)
        if epoch % DEG_FREQ == 0:
            update_pheromones(pheromones, paths, 2)
    
    # Then probably next ACO iteration:
        # Generate 5 more population, evaluate
        # Add to existing population, potentially trim
    
   
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 10:24:38 2021

@author: janneke
"""

import numpy as np
import torch
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
import pandas as pd
import os
from mdrnn import MDRNN
import tqdm as tqdm
import math
import matplotlib.pyplot as plt


class Pheromones():
    """
    Pheromones class. Stores the pheromone values for each node.
    
    Attributes may depend on number/type of layers.
    
    Atributes
    ---------
    input : pheromone levels for the input nodes
    m1 : pheromones levels for the first layer
    m2 : pheromone levels for the second layer
    m2_red : DESCRIPTION.
    """
    
    def __init__(self, n_inputs, n_outputs, max_pheromone=10):
        self.input = np.ones(n_inputs)
        self.m1 = np.ones((n_inputs * 4, n_inputs))
        self.m2 = np.ones((n_outputs * 4, n_inputs))
        self.m2_red = np.ones((n_outputs * 4, n_outputs))
        self.pheromones = {"m1":self.m1, "m2": self.m2, "m2_red": self.m2_red}
        self.max_pheromone = max_pheromone
         
    def update(self, paths, action, kind="m1"):
        """
        Update pheromones along specific connections.
        
        Parameters
        ----------
        paths : Paths object
            Paths object containing the currently active paths
        action : int
            Specifies to reward (0), penalize (1) or degrade (2) the topology 
            specified by the Paths object
        """
        
        pher = self.pheromones[kind]
        paths = paths.paths[kind]
        for i in range(0, len(paths)):
            for j in range(0, len(paths[i])):
                if action == 2:  # degrade
                        pher[i,j] *= 0.9
                        
                if paths[i,j]:
                    if action == 0:  # reward
                        pher[i,j] = min(pher[i,j] * 1.15, 
                                            self.max_pheromone)
                    elif action == 1:  # penalize
                        pher[i,j] *= 0.85


class Paths():
    def __init__(self, n_inputs, n_outputs, pheromones, ants=4):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.input = np.zeros(n_inputs, dtype=bool)
        self.paths = {"m1": np.zeros((self.n_inputs * 4, self.n_inputs), dtype=bool),
                      "m2": np.zeros((self.n_outputs * 4, self.n_outputs), dtype=bool), 
                      "m2_red": np.zeros((self.n_outputs * 4, self.n_outputs), dtype=bool)}
        self.ants = ants
        # Create pheromones object
        self.pheromones = pheromones
    
    def _flatten_reshape(self, paths, pheromones):
        flatpath = paths.flatten()
        flat_pheromones = pheromones.flatten()
        flat_pheromones = flat_pheromones / sum(flat_pheromones)
        chosen_paths = np.random.choice(len(flatpath), self.ants, 
                                        p=flat_pheromones)
        
        for i in chosen_paths:
            flatpath[i] = True
        
        return flatpath.reshape(paths.shape)
        
    
    def general_ants(self, kind="m1"):
        """General ant function"""
        paths = self.paths[kind]
        pheromones = self.pheromones.m1
        self.paths[kind] = self._flatten_reshape(paths, pheromones)
        mask = self.paths[kind]
        return torch.from_numpy(mask)
        

    def get_m1(self) -> torch.Tensor:
        """Get mask without data reduction (n_in == n_out).
        Returns:
            Tensor of size (n_inputs*4, n_outputs)
        """
        return self.general_ants("m1")

    def get_m2(self, reduce=False) -> torch.Tensor:
        """Get mask with data reduction.
        Kwargs:
            reduce (boolean):
                Whether to reduce the number of dims for the output
        Returns: 
                either
            Tensor of size (n_outputs*4, n_inputs) 
                or (with reduction)
            Tensor of size (n_outputs*4, n_outputs)
        """
        # If reduction to one output, adapt dimensions
        if reduce:
            return self.general_ants("m2_red")
        # By default return "regular" mask
        return self.general_ants("m2")
        
    
def prune_cell_m1(layer, paths) -> None:
    """Prune M1 cell.
    
    Shape of contents entry is 4*hiddens, n_input + n_output
    """
    # Copy names, contents
    names = [name for (name, content) in layer.named_parameters() 
             if "weight" in name]
    contents = [content for (name, content) in layer.named_parameters() 
                if "weight" in name]
    
    # Do we also want to prune the biases? And do we even need contents?
    
    # Pruning step
    for name, content in zip(names, contents):
        prune.custom_from_mask(layer, name=name, mask=paths.get_m1())


def prune_cell_m2(layer, paths) -> None:
    """Prune M2 cell. Since M2 cells only have one output, the paths 
    are adapted accordingly."""
    # Copy names, contents
    names = [name for (name, content) in layer.named_parameters() 
             if "weight" in name]
    contents = [content for (name, content) in layer.named_parameters() 
                if "weight" in name]
    # Pruning step
    for name, content in zip(names, contents):
        if "hh" in name:  # hh layer has 4*n_out, n_out dims
            prune.custom_from_mask(layer, name=name, 
                                   mask=paths.get_m2(True))
        else:  # ih layer has 4*n_out, n_in dims
            prune.custom_from_mask(layer, name=name, mask=paths.get_m2())


def prune_layer(model, paths, n_inputs, n_outputs):
    """
    Prunes the rnn layer of a model
    
    params: 
    model: model with to-be-pruned layer
    n_inputs: number of inputs
    n_outputs: number of outputs
    """

    # Prune the cells
    prune_cell_m1(model.rnn, paths)
    prune_cell_m2(model.rnn, paths)

    
def load_data(index):
    '''
    loads the NGAFID dataset.
    
    Only takes three features currently and disregards unique flights
    Consideration: Ignore first minute of data since the FDR is still booting
    and measurements may be unreliable.
    
    Outputs: training and test data consisting of len(data) - 1 features used
    to predict the last feature
    '''
    
    data_path = 'data'
    flights = ['C172', 'C182', 'PA28', 'PA44', 'SR20']
    
    training_data = None
    test_data = None
    
    # For each flight read the csv and concatenate it to data
    for flight in flights:
        
        # Read data but skip initial spaces and comments
        df = pd.read_csv(
            os.path.join(data_path, flight, 'log_110812_095915_KCKN.csv'), 
            comment='#',
            skipinitialspace=True)
        
        # Used features:
        df = df[['E1 FFlow', 'E1 CHT1', 'E1 EGT1']]
        
        # Use flight SR20 for test data, rest for training
        if flight == flights[index]:
            test_data = torch.Tensor(df.values)
        else:
            # Concatenate data
            if training_data is None:
                training_data = torch.Tensor(df.values)    
            training_data = torch.cat((training_data, torch.Tensor(df.values)))
    
    # Print general information
    print("\nNumber of features: {}, number of training samples: {}, number of \
test samples: {}".format(training_data.shape[1], training_data.shape[0], 
                         test_data.shape[0]))
    
    return training_data, test_data


def train(model, training_data, n_epochs, batch_size=16, lr=0.01):
    
    train_loader = DataLoader(training_data, batch_size=batch_size, 
                              shuffle=True)
    
    # Define Loss, Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(1, n_epochs + 1):
        for batch in train_loader:
            # Unsqueeze to make dimensions work in the model; it assumes the 
            # data has some sequence length, 
            # but in our case that's just 1 I think
            batch = batch.unsqueeze(0)

            # Extract target
            data, target = torch.split(batch, batch.shape[-1] - 1, dim=-1)
            optimizer.zero_grad()

            mus, sigmas, logpi = model(data)
            loss = model.loss(target, logpi, mus, sigmas)
            loss.backward()
            optimizer.step()
    return loss
    
def test(model, test_data, batch_size=16):
    
    test_loader = DataLoader(test_data, batch_size=batch_size)
    loss = []
    
    for batch in test_loader:
        
        batch = batch.unsqueeze(0)
        data, target = torch.split(batch, batch.shape[-1] - 1, dim=-1)
        
        with torch.no_grad():
            mus, sigmas, logpi = model(data)
            loss.append(model.loss(target, logpi, mus, sigmas))
    model.fitness = np.average(loss)


def ACO(aco_iterations, n_inputs, n_outputs, n_hiddens, pheromones, 
        training_data, test_data, n_gaussians):
    """
    Run the ACO algorithm for aco_iterations iterations.
    """
    # Store data:
    avg_train_losses = []
    avg_test_losses = []
    
    # Hyper parameters
    n_models = 2
    deg_freq = 5
    batch_size = 16
    n_epochs = 10
    lr = 0.01
    
    # Initialize population
    population = []
    cur_best = math.inf
    for iteration in tqdm.tqdm(range(aco_iterations)):
        # Generate paths for the models
        
        train_losses = []
        test_losses = []
        
        for n in range(n_models):
            
            # Define model and paths
            model = MDRNN(n_inputs, n_outputs, n_hiddens, n_gaussians)
            paths = Paths(n_inputs, n_hiddens, pheromones)
                        
            # Prune the model
            prune_layer(model, paths, n_inputs, n_hiddens)
         
            # Training loop 
            loss = train(model, training_data, n_epochs, batch_size, lr)
            print("\nTrain loss for model {}: {:.4f}".format(n+1, loss.item()))
            
            # Update fitness
            test(model, test_data, batch_size)
            print("Test loss for model {}: {}".format(n+1, model.fitness))
            
            # Save losses:
            train_losses.append(loss.item())
            test_losses.append(model.fitness)
            
            # Add model to population
            population.append((model, paths))
        
        # Update pheromones
        for model, paths in population:
            if model.fitness < cur_best:  # Reward
                torch.save(model.state_dict(), 'model_weights.pth')
                cur_best = model.fitness
                pheromones.update(paths, 0)
            else:  # Punishment 
                pheromones.update(paths, 1)
            if iteration % deg_freq == 0:  # Decay step
                pheromones.update(paths, 2)
        
        avg_train_losses.append(np.average(train_losses))
        avg_test_losses.append(np.average(test_losses))
    
    return avg_train_losses, avg_test_losses

n_inputs = 2
n_outputs = 1
n_hiddens = 2
n_gaussians = 5
n_iterations = 10
    
train_losses = []
test_losses = []
# Apply cross-validation
for i in range(5):
    # Initialize pheromones storage
    pheromones = Pheromones(n_inputs, n_hiddens)
    # Load data
    train_data, test_data = load_data(i)
    # Run ACO
    avg_train_losses, avg_test_losses = ACO(n_iterations, n_inputs, n_outputs, 
                                    n_hiddens, pheromones, train_data, 
                                    test_data, n_gaussians)
    
    train_losses.append(avg_train_losses)  # shape (5, n_iter)
    test_losses.append(avg_test_losses)
    

plt.title("Train loss over time")
plt.plot(np.average(train_losses, axis=0))
plt.show()
plt.title("Test loss over time")
plt.plot(np.average(test_losses, axis=0))
plt.show()

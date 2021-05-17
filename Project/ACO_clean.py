#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 10:24:38 2021

@author: janneke
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
import pandas as pd
import os
from mdrnn import MDRNN


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
        self.max_pheromone = max_pheromone
         
    def update(self, paths, action):
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
        for i in range(1, len(paths.input)):
            if paths.input[i] == 1:
                if action == 0:  # reward
                    self.input[i] = min(self.input[i] * 1.15, 
                                        self.max_pheromone)
                elif action == 1:  # penalize
                    self.input[i] *= 0.85
                elif action == 2:  # degrade
                    self.input[i] *= 0.9
                    
        for i in range(1, len(paths.m1)):
            for j in range(1, len(paths.m1[i])):
                if paths.m1[i][j] == 1:
                    if action == 0:  # reward
                        self.m1[i][j] = min(self.m1[i][j] * 1.15, 
                                            self.max_pheromone)
                    elif action == 1:  # penalize
                        self.m1[i][j] *= 0.85
                    elif action == 2:  # degrade
                        self.m1[i][j] *= 0.9


class Paths():
    def __init__(self, n_inputs, n_outputs, ants=256):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.input = np.zeros(n_inputs, dtype=bool)
        self.m1 = np.zeros((n_inputs * 4, n_inputs), dtype=bool)
        self.m2 = np.zeros((n_outputs * 4, n_inputs), dtype=bool)
        self.m2_red = np.zeros((n_outputs * 4, n_outputs), dtype=bool)
        self.ants = ants
        # Create pheromones object
        self.pheromones = Pheromones(n_inputs, n_outputs)
        
    def general_ants(self, kind="m1"):
        """General ant function"""
        if kind == "m1":
            paths = np.zeros((self.n_inputs * 4, self.n_inputs), dtype=bool)
            pheromones = self.pheromones.m1
        elif kind == "m2":
            paths = np.zeros((self.n_outputs * 4, self.n_inputs), dtype=bool)
            pheromones = self.pheromones.m2
        else:
            paths = np.zeros((self.n_outputs * 4, self.n_outputs), dtype=bool)
            pheromones = self.pheromones.m2_red
        
        flatpath = paths.flatten()
        flat_pheromones = pheromones.flatten()
        flat_pheromones = flat_pheromones / len(flat_pheromones)
        chosen_paths = np.random.choice(len(flatpath), self.ants, 
                                        p=flat_pheromones)
        
        for i in chosen_paths:
            flatpath[i] = True
        
        return torch.from_numpy(flatpath.reshape(paths.shape))
        
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
            return self.general_ants("m2r")
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
                                   mask=paths.get_m2(reduce=True))
        else:  # ih layer has 4*n_out, n_in dims
            prune.custom_from_mask(layer, name=name, mask=paths.get_m2())


def prune_layer(model, n_inputs, n_outputs):
    """
    Prunes the rnn layer of a model
    
    params: 
    model: model with to-be-pruned layer
    n_inputs: number of inputs
    n_outputs: number of outputs
    """
    
    # Hardcoded for now, no clue how to do it properly
    paths = Paths(n_inputs, n_outputs)
    
    # Prune the cells
    prune_cell_m1(model.rnn, paths)
    prune_cell_m2(model.rnn, paths)

    
def load_data():
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
        if flight == 'SR20':
            test_data = torch.Tensor(df.values)
        else:
            # Concatenate data
            if training_data is None:
                training_data = torch.Tensor(df.values)    
            training_data = torch.cat((training_data, torch.Tensor(df.values)))
    
    # Print general information
    print("Number of features: {}, number of training samples: {}, number of \
test samples: {}".format(training_data.shape[1], training_data.shape[0], 
                         test_data.shape[0]))
    
    return training_data, test_data


def train(model, training_data, batch_size=16):
    # Define hyperparameters
    n_epochs = 100
    lr = 0.01
    
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
        
        if epoch % 10 == 0:
            print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
            print("Loss: {:.4f}".format(loss.item()))
    
    
def test(model, test_data, batch_size=16):
    
    test_loader = DataLoader(test_data, batch_size=batch_size)
    loss = 0
    
    for batch in test_loader:
        
        batch = batch.unsqueeze(0)
        data, target = torch.split(batch, batch.shape[-1] - 1, dim=-1)
        
        with torch.no_grad():
            mus, sigmas, logpi = model(data)
            loss += model.loss(target, logpi, mus, sigmas)


def ACO(aco_iterations):
    """
    Run the ACO algorithm for aco_iterations iterations.
    """
    # Data parameters
    n_inputs = 2
    n_outputs = 1
    n_hiddens = 2  # should equal n_inputs
    
    # Hyper parameters
    n_gaussians = 1  # is this dependent on other factors?
    n_models = 1  # 1 for now, actually 10
    deg_freq = 5
    batch_size = 16
    
    # Initialize population
    population = []
    cur_best = -1
    # Generate pheromones
    pheromones = None 
    
    for iteration in range(aco_iterations):
        # Generate paths for the models
        paths = None
        
        for _ in range(n_models):
            training_data, test_data = load_data()
            model = MDRNN(n_inputs, n_outputs, n_hiddens, n_gaussians)
            
            # Prune the model
            # Loop layers
            prune_layer(model, n_inputs, n_hiddens)
            
            # Training loop 
            train(model, training_data, batch_size)
            
            # Update fitness
            test(model, test_data, batch_size)
            
            # Add model to population
            population.append(model)
        
        # Update pheromones
        for model in population:
            if model.fitness > cur_best:  # Reward
                cur_best = model.fitness
                pheromones.update(paths, 0)
            else:  # Punishment 
                pheromones.update(paths, 1)
            if iteration % deg_freq == 0:  # Decay step
                pheromones.update(paths, 2)
                 
                 
ACO(1)

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
import torch.nn.functional as F
import torch.optim as optim
import math
import random

N_ANTS = 5
N_INPUTS = 16
MAX_PHEROMONE = 10
DEG_FREQ = 2


class Pheromones():
    def __init__(self, n_inputs, n_outputs):
        self.input = np.ones(n_inputs)
        self.m1 = np.ones((n_inputs*4, n_inputs))
        self.m2 = np.ones((n_outputs*4, n_inputs))
        self.m2_red = np.ones((n_outputs*4, n_outputs))
        
        
    def update(self, paths, action):
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
        """
        for i in range(1, len(paths.input)):
            if paths.input[i] == 1:
                if action == 0:  # reward
                    self.input[i] = min(self.input[i] 
                                        * 1.15, MAX_PHEROMONE)
                elif action == 1:  # penalize
                    self.input[i] *= 0.85
                elif action == 2:  # degrade
                    self.input[i] *= 0.9
                    
        for i in range(1, len(paths.m1)):
            for j in range(1, len(paths.m1[i])):
                if paths.m1[i][j] == 1:
                    if action == 0:
                        self.m1[i][j] = min(self.m1[i][j]
                                            * 1.15, MAX_PHEROMONE)
                    elif action == 1:
                        self.m1[i][j] *= 0.85
                    elif action == 2:
                        self.m1[i][j] *= 0.9


class Paths():  # Maybe make paths_m1 and paths_m2
    def __init__(self, n_inputs, n_outputs):
        self.input = np.zeros(n_inputs, dtype=bool)
        self.m1 = np.zeros((n_inputs*4, n_inputs), dtype=bool)
        self.m2 = np.zeros((n_outputs*4, n_inputs), dtype=bool)
        self.m2_red = np.zeros((n_outputs*4, n_outputs), dtype=bool)
        self.pheromones = Pheromones(n_inputs, n_outputs)
        self.ants = 256
    
    def ants_m1(self):
        for ant in range(self.ants):
            # For ant in ants get random coordinates
            row = np.random.randint(len(self.m1))
            col = np.random.randint(len(self.m1[row]))
            item = self.pheromones.m1[row, col]
            decision = np.random.rand()
            if decision < item:
                self.m1[row, col] = True

    def ants_m2(self):
        for ant in range(self.ants):
            # For ant in ants get random coordinates
            row = np.random.randint(len(self.m2))
            col = np.random.randint(len(self.m2[row]))
            item = self.pheromones.m2[row, col]
            decision = np.random.rand()
            if decision < item:
                self.m2[row, col] = True
    
    def get_m1(self) -> torch.Tensor:
        """Get mask without data reduction (n_in == n_out).
        Returns:
            Tensor of size (n_inputs*4, n_outputs)
        """
        randoms = np.random.rand(self.m1.shape[0], self.m1.shape[1])
        return torch.from_numpy(randoms < self.pheromones.m1)
    
    def get_m2(self, reduce=False) -> torch.Tensor:
        """Get mask with data reduction.
        Kwargs:
            reduce (boolean):
                Whether to reduce the number of dims for the output
        Returns: 
            Tensor of either size (n_outputs*4, n_inputs) 
                or (with reduction)
            Tensor of size (n_outputs*4, n_outputs)
        """
        # If reduction to one output, adapt dimensions
        if reduce:
            randoms = np.random.rand(self.m2_red.shape[0], self.m2_red.shape[1])
            return torch.from_numpy(randoms < self.pheromones.m2_red)

        # By default return "regular" mask
        randoms = np.random.rand(self.m2.shape[0], self.m2.shape[1])  
        return torch.from_numpy(randoms < self.pheromones.m2)
        
    
def prune_cell_m1(layer, paths) -> None:
    """Prune M1 cell."""
    # Copy names, contents
    names = [name for (name, content) in layer.named_parameters()]
    contents = [content for (name, content) in layer.named_parameters() 
                if "weight" in name]
    
    # Pruning step
    for name, content in zip(names, contents):
        prune.custom_from_mask(layer, name=name, mask=paths.get_m1())


def prune_cell_m2(layer, paths) -> None:
    """Prune M2 cell. Since M2 cells only have one output, the paths 
    are adapted accordingly."""
    # Copy names, contents
    names = [name for (name, content) in layer.named_parameters()]
    contents = [content for (name, content) in layer.named_parameters() 
                if "weight" in name]
    
    # Pruning step
    for name, content in zip(names, contents):
        if "hh" in name:  # hh layer has 4*n_out, n_out dims
            prune.custom_from_mask(layer, name=name, \
                                   mask=paths.get_m2(reduce=True))
        else:  # ih layer has 4*n_out, n_in dims
            prune.custom_from_mask(layer, name=name, mask=paths.get_m2())

paths = Paths(16, 4).ants_m1()
m1_cell = nn.LSTMCell(16, 16)
m2_cell = nn.LSTMCell(16, 4)
prune_cell_m1(m1_cell, paths)
prune_cell_m2(m2_cell, paths)

def train():
    pass

def test():
    pass


def ACO(aco_iterations):
    """
    """
    # Initialize population
    population = []
    cur_best = -1
    # Generate pheromone s
    pheromones = None 
    
    for iteration in aco_iterations:
        # Generate a new set of models
        models = ["LSTM for l in number_models"]
        
        # Generate paths for the models
        paths = None
        
        for model in models:
            # Prune the model
            # Loop layers
            prune_layer(model)
            
            # Training loop 
            train(model)
            
            # Update fitness
            test(model)
        
        # Add models to population
        [population.append(model) for model in models]
        
        # Update pheromones
        for model in population:
            if model.fitness > cur_best:  # Reward
                cur_best = model.fitness
                pheromones.update(paths, 0)
            else:  # Punishment 
                pheromones.update(paths, 1)
            if iteration % DEG_FREQ == 0:  # Decay step
                 pheromones.update(paths, 2)
        
            
            
            
            
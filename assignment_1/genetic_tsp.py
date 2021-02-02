#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 08:23:30 2021

@author: janneke
"""
import os
import math
import numpy as np
#print(os.getcwd())
#os.chdir(os.environ["HOME"]+"/natural-computing/assignment-1/")

def dist(node1, node2): 
    """Euclidean distance measure for nodes in TSP graph"""
    return math.sqrt((node1[0]-node2[0])**2+(node1[1]-node2[1])**2)

def fitness(ind):
    """Fitness of individual ind, computed by summing Euclidean distances between
    pairs of nodes."""
    total_dist = 0
    for i in range(1,  len(ind)):
        total_dist += dist(ind[i-1], ind[i])
    return total_dist

def permute(ind):
    "Switch genes of individual at random indices"
    length = len(ind)
    p = np.random.randint(0, length)
    q = np.random.randint(0, length)
    v = ind[q]   # Switch genes at indices
    ind[q] = ind[p]
    ind[p] = v
    return ind

def crossover(ind1, ind2):
    """Perform random crossover operation between ind1 and ind2"""
    #@TODO: implement looparound
    length = len(ind1)
    p = np.random.randint(1, length-1)
    q = np.random.randint(p, length-1)   # Ensure q > p
    
    cross1 = ind2[p:q] # Slice crossover section
    cross2 = ind1[p:q] 
    
    add1 = [n for n in ind2 if n not in cross1]   # Get all genes in parent that are not in crossover part
    add2 = [n for n in ind1 if n not in cross2]   # Swap crossover section
    
    child1 = np.vstack((add1[:p], cross1, add1[p:]))  # Add parent genes to crossover in order
    child2 = np.vstack((add2[:p], cross2, add2[p:])) 
    
    return child1, child2
    
def n_opt(solution, n=2):
    """Apply n-opt to the solution. This is effectively slicing the full path into n+1 pieces, and then
    recombining those pieces. If the full path is a cycle, then two slices need to be made, if it is not then 
    one slice suffices."""
    
    p = np.random.randint(len(solution)-1)
    part1 = solution[:p]
    part2 = solution[p:]
    arr = np.array([part1, part2], dtype="object")
    perm = np.random.permutation(n+1)
    
    # Decide how to recombine
    n_opt = arr[perm]
    n_opt = np.hstack(n_opt)
    
    if fitness(n_opt) >= fitness(solution):
        return n_opt    
    
    return solution
    
with open("file-tsp.txt") as file:
    nodes = [file.readline().strip().split() for line in file]
    nodes = np.array([np.array(list(map(float, node))) for node in nodes])

def run(problem, gen_size = 2, pop_size = 10, n_gen = 1000):
    candidates = []
    for i in range(pop_size):
        candidates.append(np.random.permutation(problem))
   
    for n in range(n_gen):
        c_sort = sorted(candidates, key=lambda x: fitness(x), reverse=True)
        parents = c_sort[:gen_size]
        crossover(parents[0], parents[1])
    
run(nodes)
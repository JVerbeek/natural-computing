#!/usr/bin/env python
# coding: utf-8

# # Ensemble Learning
# 
# ## Exercise 2.b

# In[7]:


import numpy as np
from scipy.special import comb


# In[22]:


def maj_vote(c, p):
    """Function to determine the success chance by majority vote for a jury of c with success chance p"""
    # d is the number of required successes
    d = c / 2
    if c % 2 == 0:
        d += 0.5
    d += 0.5
    d = int(d)

    result = 0
    for _ in range(c - d + 1):
        result += comb(c, d) * p ** d * (1 - p) ** (c - d)
        d += 1

    return result

c = 31
p = 0.6
print(maj_vote(c, p))


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # Ensemble Learning
# 
# ## Exercise 2
# 
# ## b

# In[7]:


import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt


# In[96]:


def maj_vote(c, p):
    """Function to determine the success chance by majority vote for a jury of c with success chance p"""
    if c == 1:
        return p
    
    # d is the number of required successes
    d = c / 2
    if c % 2 == 0:
        d += 0.5
    d += 0.5
    d = int(d)

    result = 0
    for _ in range(d, c + 1):
        result += comb(c, d) * p ** d * (1 - p) ** (c - d)
        d += 1

    return result

c = 31
p = 0.6
print(maj_vote(c, p))


# ## c

# In[98]:


def plot(n, p):
    results = np.empty(50)
    for i, c in enumerate(range(1, 51)):
            results[i] = maj_vote(c, p)
    plt.subplot(2, 2, n)
    plt.plot(np.arange(1, 51), results)
    plt.title("Majority vote success rate for p = {}".format(p))
    plt.xlabel("Jury size")
    plt.ylabel("p")
    plt.ylim(-0.05, 1.05)
    
plt.figure(figsize=(12, 8))
for n, p in enumerate([0.2, 0.4, 0.6, 0.8]):
    plot(n + 1, p)
plt.tight_layout()
plt.show()


# ## d, e

# In[99]:


print(maj_vote(1, 0.85))
print(maj_vote(3, 0.75))
print(maj_vote(31, 0.6))

print(maj_vote(25, 0.6))


# ## Exercise 3
# ## b

# In[120]:


without = 0
with_ = 0
c = 10
p = 0.6
strong_p = 0.75

d = c / 2
if c % 2 == 0:
    d += 0.5
d += 0.5
d = int(d)

for i in range(d - 1, c + 1):
    if i >= d:
        without += comb(c, i) * p ** i * (1 - p) ** (c - i) * (1 - strong_p)
    with_ += comb(c, i) * p ** i * (1 - p) ** (c - i) * strong_p    
    
result = without + with_
print(result)


# In[106]:


import numpy as np
import matplotlib.pyplot as plt
import poisson_binomial as pb
import math 

def fact(x):
    """Compute a factorial"""
    if x == 0:
        return 1
    else:
        return x*fact(x-1)

def vote(c, p):
    """Implementation of general formula for making the correct decision using majority vote"""
    result = 0
    if c == 1:
        return p
    for k in range(int(c/2), c):
        result += (fact(c)/(fact(k)*fact(c-k))) * (p**k) *((1-p)**(c-k))
    return result

ps_even = []
ps_odd = []
for i in range(1, 101, 2):
    ps_even.append(vote(i, 0.75))
plt.plot(ps_even)
plt.show()

def weighted_vote_poibin(c, p, w):
    new_p = []
    for i in range(len(w)):
        new_p += w[i]*[p[i]]
    poibin = pb.PoissonBinomial(new_p)
    #print(vote(len(new_p), np.average(new_p)))
    return poibin.x_or_more(int(len(new_p)/2))

def weighted_vote_avg(c, p, w):
    new_p = []
    for i in range(len(w)):
        new_p += w[i]*[p[i]]
    return vote(len(new_p), np.average(new_p))

#print(ensemble_vote(6, [0.75, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]))

wv_poibin = []
wv_avg = []
for i in range(0, 90, 1):
    wv_poibin.append(weighted_vote_poibin(11, [0.75, 0.6],  [i, 10]))
    print(i/(i+10))
    wv_avg.append(weighted_vote_avg(11, [0.75, 0.6], [i, 10]))
plt.plot(wv_poibin)
plt.plot(wv_avg)


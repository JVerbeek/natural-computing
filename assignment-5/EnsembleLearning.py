#!/usr/bin/env python
# coding: utf-8

# # Ensemble Learning
# 
# ## Exercise 2
# 
# ## b

# In[5]:


import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
import poisson_binomial as pb


# In[6]:


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

# In[7]:


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

# In[33]:


print(maj_vote(1, 0.85))
print(maj_vote(3, 0.75))
print(maj_vote(31, 0.6))

print(maj_vote(25, 0.6))


# ## Exercise 3
# ## a

# In[34]:


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

new_p = (p * c + strong_p) / 11
print(maj_vote(11, new_p))

for i in range(d - 1, c + 1):
    if i >= d:
        without += comb(c, i) * p ** i * (1 - p) ** (c - i) * (1 - strong_p)
    with_ += comb(c, i) * p ** i * (1 - p) ** (c - i) * strong_p    
    
result = without + with_
print(result)


# ## b

# In[37]:


def weighted_vote(c, ps, w):
    p, strong_p = ps
    without = 0
    with_ = 0
    
    if w == 0:
        return maj_vote(c  - 1, p)
    
    d = c / 2
    if c % 2 == 0:
        d += 0.5
    d += 0.5
    d = int(d)
    
    new_p = 1 - (1 - strong_p) ** w
    
    for i in range(d - 1, c + 1):
        if i >= d:
            without += comb(c, i) * p ** i * (1 - p) ** (c - i) * (1 - new_p)
        with_ += comb(c, i) * p ** i * (1 - p) ** (c - i) * new_p

    return without + with_    


c = 11
p = (0.6, 0.75)
n = 50
results = np.empty(n)
for i, w in enumerate(range(0, n)):
    results[i] = weighted_vote(c, p, w / 10)
plt.plot(np.arange(n) / 10, results)
plt.xlabel("Weight")
plt.ylabel("Success chance")
plt.title("Weighted majority vote")
plt.show()


# In[69]:


def AdaBoost(p):
    N = 11
    w = np.full(N, 1/N)
    M = 3
    np.random.seed(37)
    
    for m in range(M):
        correct = np.zeros(N)
        for i in range(N):
            if np.random.rand() < p[i]:
                correct[i] = 1
        
        error = np.sum(w * correct) / np.sum(w)
        print("Error: {}".format(error))
        alpha_m = np.log((1 - error) / error)
        print("Alpha: {}".format(alpha_m))
        w *= np.exp(alpha_m * correct)
        print("w: {}".format(w))
    return w
        
        
p = np.append(np.repeat(0.6, 10), 0.75)
print(AdaBoost(p))


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


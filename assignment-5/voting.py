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
    print(vote(i, 0.75))
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
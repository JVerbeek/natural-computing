#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 16:16:12 2021

@author: janneke
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, GridSearchCV

data = pd.read_csv("~/natural-computing/covtype.data", header = None)
X = data.iloc[:,:30].values
y = data.iloc[:,54].values

plt.hist(y)

FOLDS = 3
cv = KFold(FOLDS)

#accuracies = np.zeros((FOLDS, 8))
#for d in range(1, 31, 5):
#    print("woop")
#    for i, (train, test) in enumerate(kf.split(X, y)):
#        X_train, X_test = X[train], X[test]
#        y_train, y_test = y[train], y[test]
#        clf = RandomForestClassifier(max_depth=d, min_samples_split = 2)
#        clf.fit(X_train, y_train)
#        y_pred = clf.predict(X_test)
#        accuracies[i, int(d/5)] = accuracy_score(y_test, y_pred)
#        print(accuracy_score(y_test, y_pred), d, i)

#print(np.mean(accuracies, axis=0))

"""
clf_ = RandomForestClassifier()
depths = range(1, 31, 3)
estimators = [20, 30, 40, 50, 60, 70, 80, 90, 100]
#estimators = [10, 100]
params = {"max_depth": depths, "n_estimators": estimators}
print("Starting...")
clf = GridSearchCV(clf_, params, verbose=3, n_jobs=-1)
clf.fit(X, y)
print("Done!")
scores = clf.cv_results_["mean_test_score"]
print(scores)
scores = np.array(scores).reshape(len(depths), len(estimators))

for ind, i in enumerate(depths):
    plt.plot(estimators, scores[ind], "-o", label=f"depth: {i}")
plt.legend()
plt.title("Mean accuracy vs. # estimators per depth")
plt.xlabel("# estimators")
plt.ylabel("Mean score")
plt.show()
"""
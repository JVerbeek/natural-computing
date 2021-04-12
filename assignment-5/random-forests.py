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
from sklearn.model_selection import KFold

data = pd.read_csv("~/natural-computing/covtype.data", header = None)
X = data.iloc[:,:10].values
y = data.iloc[:,54].values
FOLDS = 5
kf = KFold(FOLDS)

accuracies = np.zeros((FOLDS, 8))
for d in range(1, 41, 5):
    print("woop")
    for i, (train, test) in enumerate(kf.split(X, y)):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        clf = RandomForestClassifier(max_depth=d, min_samples_split = 2)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracies[i, int(d/40)] = accuracy_score(y_test, y_pred)
        print(accuracy_score(y_test, y_pred), d, i)

print(np.mean(accuracies, axis=0))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 23:27:03 2019

@author: dadajonjurakuziev
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
missing_values = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=0)
missing_values = missing_values.fit(X[:, 1:3])
X[:, 1:3] = missing_values.transform(X[:, 1:3])
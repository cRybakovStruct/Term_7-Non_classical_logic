# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 10:48:04 2019

@author: Vladimir
"""

import mglearn
import matplotlib.pyplot as plt
import numpy as np

# In[8]: 
from sklearn.datasets import load_boston 
boston = load_boston() 
print("форма массива data для набора boston: {}".format(boston.data.shape)) 

# In[9]: 
X, y = mglearn.datasets.load_extended_boston() 
print("форма массива X: {}".format(X.shape))

# In[10]: 
mglearn.plots.plot_knn_classification(n_neighbors=1) 

# In[11]: 
mglearn.plots.plot_knn_classification(n_neighbors=3) 

# In[12]: 
from sklearn.model_selection import train_test_split 
X, y = mglearn.datasets.make_forge() 
 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0) 

# In[13]: 
from sklearn.neighbors import KNeighborsClassifier 
clf = KNeighborsClassifier(n_neighbors=3)

# In[14]: 
clf.fit(X_train, y_train) 

# In[15]: 
print("Прогнозы на тестовом наборе: {}".format(clf.predict(X_test))) 

# In[16]: 
print("Правильность на тестовом наборе: {:.2f}".format(clf.score(X_test, y_test))) 

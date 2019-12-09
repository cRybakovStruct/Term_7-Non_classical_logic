# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:58:46 2019

@author: Vladimir
"""
#import numpy as np 
#import matplotlib.pyplot as plt 
#import pandas as pd 
#import mglearn 
## %matplotlib inline 
#from sklearn.model_selection import train_test_split 
#from sklearn.datasets import load_breast_cancer 
#from sklearn import tree 
#from sklearn.tree import export_graphviz 
#cancer = load_breast_cancer() 
#X_train, X_test, y_train, y_test = train_test_split( 
#    cancer.data, cancer.target, stratify=cancer.target, random_state=42) 
#clf = tree.DecisionTreeClassifier(max_depth=4, random_state=0) 
#clf = clf.fit(X_train, y_train) 
# 
## import pydotplus 
#dot_data = tree.export_graphviz(clf, out_file=None) 
#
#from sklearn.tree import DecisionTreeClassifier 
 
#cancer = load_breast_cancer() 
#X_train, X_test, y_train, y_test = train_test_split( 
#    cancer.data, cancer.target, stratify=cancer.target, random_state=42) 
#tree = DecisionTreeClassifier(random_state=0) 
#tree.fit(X_train, y_train) 
#print("Правильность на обучающем наборе: {:.3f}".format(tree.score(X_train, y_train))) 
#print("Правильность на тестовом наборе: {:.3f}".format(tree.score(X_test, y_test))) 
#
#tree = DecisionTreeClassifier(max_depth=4, random_state=0) 
#tree.fit(X_train, y_train) 
# 
#print("Правильность на обучающем наборе: {:.3f}".format(tree.score(X_train, y_train))) 
#print("Правильность на тестовом наборе: {:.3f}".format(tree.score(X_test, y_test))) 

#=========================================================
#
#from sklearn.ensemble import RandomForestClassifier 
#from sklearn.datasets import make_moons 
# 
#X, y = make_moons(n_samples=100, noise=0.25, random_state=3) 
#X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, 
#                                                    random_state=42) 
# 
#forest = RandomForestClassifier(n_estimators=5, random_state=2) 
#forest.fit(X_train, y_train)
#
#fig, axes = plt.subplots(2, 3, figsize=(20, 10)) 
#for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)): 
#    ax.set_title("Дерево {}".format(i)) 
#    mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax) 
# 
#mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1, -1], 
#                                alpha=.4) 
#axes[-1, -1].set_title("Случайный лес") 
#mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train) 
#
#X_train, X_test, y_train, y_test = train_test_split( 
#    cancer.data, cancer.target, random_state=0) 
#forest = RandomForestClassifier(n_estimators=100, random_state=0) 
#forest.fit(X_train, y_train) 
# 
#print("Правильность на обучающем наборе: {:.3f}".format(forest.score(X_train, y_train)))
#print("Правильность на тестовом наборе: {:.3f}".format(forest.score(X_test, y_test))) 

#========================================================================================

import graphviz 
 
with open("tree.dot") as f: 
    dot_graph = f.read() 
graphviz.Source(dot_graph) 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import mglearn 
#%matplotlib inline 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_breast_cancer 
from sklearn import tree 
from sklearn.tree import export_graphviz 
cancer = load_breast_cancer() 
X_train, X_test, y_train, y_test = train_test_split( 
    cancer.data, cancer.target, stratify=cancer.target, random_state=42) 
clf = tree.DecisionTreeClassifier(max_depth=4, random_state=0) 
clf = clf.fit(X_train, y_train) 
 
import pydotplus 
dot_data = tree.export_graphviz(clf, out_file=None) 

graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("cancer.pdf") 

 
from IPython.display import Image 
dot_data = tree.export_graphviz(clf, out_file=None, 
                     feature_names=cancer.feature_names, 
                     class_names=cancer.target_names, 
                     filled=True, rounded=True, 
                     special_characters=True) 
graph = pydotplus.graph_from_dot_data(dot_data) 
Image(graph.create_png()) 

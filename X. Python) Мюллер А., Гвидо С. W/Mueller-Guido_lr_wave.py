# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:40:12 2019

@author: Vladimir
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import mglearn

from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features=1, noise=0.0)

# X, y = mglearn.datasets.make_wave(n_samples=100)
plt.plot(X, y, 'og')
plt.ylim(-3, 3)
plt.xlabel("Признак")
plt.ylabel("Целевая переменная")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)

print("lr.coef_: {}".format(lr.coef_[0]))
print("lr.intercept_: {}".format(lr.intercept_))

yP=lr.predict(X)

plt.plot(X, yP, '-m')
plt.grid()
k=1
x0 = np.array(X[k])
x0=x0.reshape(1, 1)
y0 = y[k]
plt.plot(x0, y0,  'ob')
y0p = lr.predict(x0)
plt.plot(x0, y0p,  'sb')

print("Правильность на обучающем наборе: {:.2f}".format(lr.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.2f}".format(lr.score(X_test, y_test)))
print("Прогнозы для тестового набора:\n{}".format(lr.predict(X_test)))
print("R^2 на тестовом наборе: {:.2f}".format(lr.score(X_test, y_test)))
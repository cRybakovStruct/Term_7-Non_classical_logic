# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 10:55:44 2019

@author: Vladimir
"""


import mglearn
import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn.linear_model import LinearRegression

area_key = 'square'
sale_key = 'V'
areas = []
sales = []

excel_data = pandas.read_excel('excel.xlsx', sheet_name='Sheet1')

for iter in range(14):
    areas.append(excel_data[area_key][iter])
    sales.append(excel_data[sale_key][iter])

#x, y = mglearn.datasets.make_wave(n_samples=100)
x = []
y = []
for area_value in areas:
    x.append([area_value])
    
for sale_value in sales:
    y.append([sale_value])
    
x = np.asarray(x)
y = np.asarray(y)

plt.plot(x, y, 'og')
plt.ylim(0, 15)
plt.xlabel('Признак')
plt.ylabel('Целевая переменная')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
lr = LinearRegression().fit(x_train, y_train)

print('lr.coef_: {}'.format(lr.coef_[0]))
print('lr.intercept_: {}'.format(lr.intercept_))

yP = lr.predict(x)
plt.plot(x, yP, '-m')
plt.grid()
k=1

x0=x[k]
x0=x0.reshape(1,1)
y0=y[k]
plt.plot(x0, y0, 'ob')
y0P=lr.predict(x0)
plt.plot(x0, y0P, 'sb')

print('Правильность на обучающем наборе: {:.2f}'.format(lr.score(x_train, y_train)))
print('Правильность на тестовом наборе: {:.2f}'.format(lr.score(x_test, y_test)))
print('Прогнозы для тестового набора:\n{}'.format(lr.predict(x_test)))
print('R^2 на тестовом наборе: {:.2f}'.format(lr.score(x_test, y_test)))
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:49:35 2019

@author: Vladimir
"""

import mglearn
import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn.linear_model import LinearRegression
import csv

with open("test.csv", "r") as f:
    rd = csv.reader(f, delimiter=';')
    list_data = [row for row in rd]
print(list_data)

nameData = list_data[0]
Data=list_data[1:]
lenData=len(Data)
Data=np.array(Data,dtype=np.float32)
print(Data)
X=Data[:,1]
X=np.reshape(X,(lenData, 1))
print(X)

Y=Data[:,2]
Y=np.reshape(Y,(lenData,1))
print(Y)

fig0, ax0 = plt.subplots()
plt.title("Исходный массив данных")
plt.plot(X, Y, 'ob')
#plt.ylim(-20, 20)
plt.xlabel("Площадь")
plt.ylabel("Доход")

from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X,Y)
YE=lr.predict(X)
plt.plot(X, YE, '-g')
plt.grid()
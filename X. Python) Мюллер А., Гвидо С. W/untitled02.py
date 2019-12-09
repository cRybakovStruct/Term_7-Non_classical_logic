# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:40:12 2019

@author: Vladimir
"""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

a=0
b=10
def f(x):
    return np.cos(-x**2/9.0)

N=101
X=np.linspace(a,b,num=N, endpoint=True)
Y=f(X)

fig1, _ = plt.subplots()
plt.plot(X,Y, '--b')
plt.xlim(a,b)
plt.xlabel("x")
plt.ylabel("y")

Nd=11
Xd=np.linspace(a,b,num=Nd, endpoint=True)
Yd=f(Xd)
plt.plot(Xd, Yd, 'ob')

f1 = interp1d(Xd, Yd, kind='linear')
Y1=f1(X)
plt.plot(X,Y1, '-k')
plt.legend(['data', 'linear'], loc='best')
plt.title('Линейная интерполяция')

f2 = interp1d(Xd,Yd,kind='cubic')
Y2=f2(X)
fig2, _ = plt.subplots()
plt.plot(X,Y, '--b')
plt.plot(X, Y2,'-m')
plt.xlim(a,b)
plt.xlabel("x")
plt.ylabel("y")
plt.title('Кубические сплайны')
plt.legend(['data', 'cubic'], loc='best')

eps1=max(abs(f1(X)-f(X)))
print('Линейная интерполяция eps=', eps1)
eps2=max(abs(f2(X)-f(X)))
print('Кубическая интерполяция eps=', eps2)




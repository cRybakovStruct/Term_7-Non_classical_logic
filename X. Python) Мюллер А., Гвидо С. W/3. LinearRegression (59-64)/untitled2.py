# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 11:47:56 2019

@author: Vladimir
"""

## In[1]:
import mglearn
import matplotlib.pyplot as plt
import numpy as np

# In[3]: 
X, y = mglearn.datasets.make_wave(n_samples=40) 
plt.plot(X, y, 'o') 
plt.ylim(-3, 3) 
plt.xlabel("Признак") 
plt.ylabel("Целевая переменная") 

# In[12]:
from sklearn.model_selection import train_test_split 
X, y = mglearn.datasets.make_forge() 
 
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)  
reg = LinearRegression()
reg.fit(X_train, y_train)
#print("%.3f" % reg.score(X_train, y_train))
#print("Coeff, intercet :/n", "%.3f" % reg.coef_, )

print("lr.coef_: {}".format(reg.coef_)) 
print("lr.intercept_: {}".format(reg.intercept_)) 

y_trainP = reg.predict(X_train)

plt.plot(X_train, y_trainP, '-m', linewidth=2)

print("%.3f" % reg.score(X_test, y_test))

x0 = X_test[3]
y0 = y_test[3]
y0p=reg.predict(x0)



#
#lr = LinearRegression().fit(X_train, y_train)
#
#print("lr.coef_: {}".format(lr.coef_)) 
#print("lr.intercept_: {}".format(lr.intercept_)) 
#
#print("Правильность на обучающем наборе: {:.2f}".format(lr.score(X_train, y_train))) 
#print("Правильность на тестовом наборе: {:.2f}".format(lr.score(X_test, y_test))) 

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:54:21 2019

@author: Vladimir
"""

import mglearn
import matplotlib.pyplot as plt
import numpy as np

#random.seed((0.1))
#codes = 3

#X, y = mglearn.datasets.make_forge()
X, y = mglearn.datasets.make_blobs(centers=2, random_state=4, n_samples=50)

X1 = X[ :, 0];
X2 = X[ :, 1];

fig1, _ = plt.subplots()
plt.plot(X1, X2, 'ob')
plt.title('fig1')
plt.xlabel("Первы признак")
plt.ylabel("Второй признак")

from sklearn.cluster import KMeans

mykmeans = KMeans(n_clusters=2, random_state=2)
mykmeans.fit(X)
c=mykmeans.cluster_centers_
y=mykmeans.labels_
print(y)
mask = y == 0
fig2,_ = plt.subplots()
plt.plot(X[mask, 0], X[mask, 1], 'ob')
plt.plot(X[~mask, 0], X[~mask, 1], 'sm')
plt.legend(['0', '1'], loc='best')
plt.scatter(c[0, 0], c[0, 1], c='b', marker='*', s=100)
plt.scatter(c[1, 0], c[1, 1], c='m', marker='*', s=100)
plt.xlabel("Первы признак")
plt.ylabel("Второй признак")

X_new = [10, 5]
plt.plot(X_new[0], X_new[1], c='k', marker='v', markersize=10)
X_new = np.reshape(X_new, (1,2))
y_pred = mykmeans.predict(X_new)

#Переписать код так, чтобы можно было задавать произвольное количестов кластеров.

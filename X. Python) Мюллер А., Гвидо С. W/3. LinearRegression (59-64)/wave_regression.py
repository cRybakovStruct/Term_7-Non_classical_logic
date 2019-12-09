# In[1]:
import mglearn
import matplotlib.pyplot as plt
import numpy as np

# In[2]:
# генерируем набор данных 
X, y = mglearn.datasets.make_forge() 
# строим график для набора данных 
#%matplotlib inline 
mglearn.discrete_scatter(X[:, 0], X[:, 1], y) 
plt.legend(["Класс 0", "Класс 1"], loc=4) 
plt.xlabel("Первый признак") 
plt.ylabel("Второй признак") 
print("форма массива X: {}".format(X.shape)) 

# In[3]:
X, y = mglearn.datasets.make_wave(n_samples=40) 
plt.plot(X, y, 'o') 
plt.ylim(-3, 3) 
plt.xlabel("Признак") 
plt.ylabel("Целевая переменная")

## In[4]: 
#from sklearn.datasets import load_breast_cancer 
#cancer = load_breast_cancer() 
#print("Ключи cancer(): \n{}".format(cancer.keys())) 
#
## In[5]: 
#print("Форма массива data для набора cancer: {}".format(cancer.data.shape)) 
#
## In[6]: 
#print("Количество примеров для каждого класса:\n{}".format( 
#      {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))})) 
#
## In[7]: 
#print("Имена признаков:\n{}".format(cancer.feature_names))
#
## In[8]: 
#from sklearn.datasets import load_boston 
#boston = load_boston() 
#print("форма массива data для набора boston: {}".format(boston.data.shape)) 
#
## In[9]: 
#X, y = mglearn.datasets.load_extended_boston() 
#print("форма массива X: {}".format(X.shape)) 
#
## In[10]: 
#mglearn.plots.plot_knn_classification(n_neighbors=1) 
#
## In[11]: 
#mglearn.plots.plot_knn_classification(n_neighbors=3) 
#
## In[12]: 
#from sklearn.model_selection import train_test_split 
#X, y = mglearn.datasets.make_forge() 
# 
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0) 
#
## In[13]: 
#from sklearn.neighbors import KNeighborsClassifier 
#clf = KNeighborsClassifier(n_neighbors=3) 
#
## In[14]: 
#clf.fit(X_train, y_train) 
#
## In[15]: 
#print("Прогнозы на тестовом наборе: {}".format(clf.predict(X_test))) 
#
## In[16]: 
#print("Правильность на тестовом наборе: {:.2f}".format(clf.score(X_test, y_test))) 
#
## In[17]: 
#fig, axes = plt.subplots(1, 3, figsize=(10, 3)) 

# In[18]: 
from sklearn.datasets import load_breast_cancer 
 
cancer = load_breast_cancer() 
X_train, X_test, y_train, y_test = train_test_split( 
    cancer.data, cancer.target, stratify=cancer.target, random_state=66) 
 
training_accuracy = [] 
test_accuracy = [] 
# пробуем n_neighbors от 1 до 10 
neighbors_settings = range(1, 11) 
 
for n_neighbors in neighbors_settings: 
    # строим модель 
    clf = KNeighborsClassifier(n_neighbors=n_neighbors) 
    clf.fit(X_train, y_train) 
    # записываем правильность на обучающем наборе 
    training_accuracy.append(clf.score(X_train, y_train)) 
    # записываем правильность на тестовом наборе 
    test_accuracy.append(clf.score(X_test, y_test)) 
 
plt.plot(neighbors_settings, training_accuracy, label="правильность на обучающем наборе") 
plt.plot(neighbors_settings, test_accuracy, label="правильность на тестовом наборе") 
plt.ylabel("Правильность") 
plt.xlabel("количество соседей") 
plt.legend() 

# In[19]: 
mglearn.plots.plot_knn_regression(n_neighbors=1) 

# In[20]: 
mglearn.plots.plot_knn_regression(n_neighbors=3) 

# In[21]: 
from sklearn.neighbors import KNeighborsRegressor 
 
X, y = mglearn.datasets.make_wave(n_samples=140) 
 
# разбиваем набор данных wave на обучающую и тестовую выборки 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0) 
 
# создаем экземпляр модели и устанавливаем количество соседей равным 3 
reg = KNeighborsRegressor(n_neighbors=3) 
# подгоняем модель с использованием обучающих данных и обучающих ответов 
reg.fit(X_train, y_train)

# In[22]: 
print("Прогнозы для тестового набора:\n{}".format(reg.predict(X_test))) 

# In[23]: 
print("R^2 на тестовом наборе: {:.2f}".format(reg.score(X_test, y_test))) 

# In[24]: 
fig, axes = plt.subplots(1, 3, figsize=(15, 4)) 
# создаем 1000 точек данных, равномерно распределенных между -3 и 3 
line = np.linspace(-3, 3, 1000).reshape(-1, 1) 
for n_neighbors, ax in zip([1, 3, 9], axes): 
    # получаем прогнозы, используя 1, 3, и 9 соседей 
    reg = KNeighborsRegressor(n_neighbors=n_neighbors) 
    reg.fit(X_train, y_train) 
    ax.plot(line, reg.predict(line)) 
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8) 
    ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8) 
     
    ax.set_title( 
        "{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format( 
            n_neighbors, reg.score(X_train, y_train), 
            reg.score(X_test, y_test))) 
    ax.set_xlabel("Признак") 
    ax.set_ylabel("Целевая переменная") 
axes[0].legend(["Прогнозы модели", "Обучающие данные/ответы", 
                "Тестовые данные/ответы"], loc="best")
    
# In[25]: 
mglearn.plots.plot_linear_regression_wave() 

# In[26]: 
from sklearn.linear_model import LinearRegression 
X, y = mglearn.datasets.make_wave(n_samples=60) 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42) 
 
lr = LinearRegression().fit(X_train, y_train) 

# In[27]: 
print("lr.coef_: {}".format(lr.coef_)) 
print("lr.intercept_: {}".format(lr.intercept_)) 

# In[28]: 
print("Правильность на обучающем наборе: {:.2f}".format(lr.score(X_train, y_train))) 
print("Правильность на тестовом наборе: {:.2f}".format(lr.score(X_test, y_test))) 

# In[29]: 
X, y = mglearn.datasets.load_extended_boston() 
 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0) 
lr = LinearRegression().fit(X_train, y_train) 

# In[30]: 
print("Правильность на обучающем наборе: {:.2f}".format(lr.score(X_train, y_train)))

# In[31]: 
from sklearn.linear_model import Ridge 
 
ridge = Ridge().fit(X_train, y_train) 
print("Правильность на обучающем наборе: {:.2f}".format(ridge.score(X_train, y_train))) 
print("Правильность на тестовом наборе: {:.2f}".format(ridge.score(X_test, y_test))) 

# In[32]: 
ridge10 = Ridge(alpha=10).fit(X_train, y_train) 
print("Правильность на обучающем наборе: {:.2f}".format(ridge10.score(X_train, y_train))) 
print("Правильность на тестовом наборе: {:.2f}".format(ridge10.score(X_test, y_test))) 

# In[33]: 
ridge01 = Ridge(alpha=0.1).fit(X_train, y_train) 
print("Правильность на обучающем наборе: {:.2f}".format(ridge01.score(X_train, y_train))) 
print("Правильность на тестовом наборе: {:.2f}".format(ridge01.score(X_test, y_test))) 

# In[34]: 
plt.plot(ridge.coef_, 's', label="Гребневая регрессия alpha=1") 
plt.plot(ridge10.coef_, '^', label="Гребневая регрессия alpha=10") 
plt.plot(ridge01.coef_, 'v', label="Гребневая регрессия alpha=0.1") 
 
plt.plot(lr.coef_, 'o', label="Линейная регрессия") 
plt.xlabel("Индекс коэффициента") 
plt.ylabel("Оценка коэффициента") 
plt.hlines(0, 0, len(lr.coef_)) 
plt.ylim(-25, 25) 
plt.legend() 

# In[35]: 
mglearn.plots.plot_ridge_n_samples()
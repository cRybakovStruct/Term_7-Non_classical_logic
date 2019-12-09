#Классификация сортов ириса (стр. 28-34)
#scikit-learn
#Классификация методом k ближайших соседей (стр. 34-37)
import numpy as np
import mglearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

iris_dataset = load_iris()
#print("Ключи iris_dataset: \n{}".format(iris_dataset.keys()))
#print(iris_dataset['DESCR'][:193] + "\n...")
#print("Названия ответов: {}".format(iris_dataset['target_names']))
#print("Названия признаков: \n{}".format(iris_dataset['feature_names']))
#print("Тип массива data: {}".format(type(iris_dataset['data'])))
#print("Форма массива data: {}".format(iris_dataset['data'].shape))
#print("Первые пять строк массива data:\n{}".format(iris_dataset['data'][:5]))
#print("Тип массива target: {}".format(type(iris_dataset['target'])))
#print("Форма массива target: {}".format(iris_dataset['target'].shape))
#print("Ответы:\n{}".format(iris_dataset['target']))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
iris_dataset['data'], iris_dataset['target'], random_state=0)
#print("форма массива X_train: {}".format(X_train.shape))
#print("форма массива y_train: {}".format(y_train.shape))
#print("форма массива X_test: {}".format(X_test.shape))
#print("форма массива y_test: {}".format(y_test.shape))
# создаем dataframe из данных в массиве X_train
# маркируем столбцы, используя строки в iris_dataset.feature_names
import pandas as pd

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# создаем матрицу рассеяния из dataframe, цвет точек задаем с помощью y_train
grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)



#knn = KNeighborsClassifier(n_neighbors=1)
knn=KNeighborsClassifier(algorithm='auto', leaf_size=30,
metric='minkowski',metric_params=None, n_jobs=1,
n_neighbors=1, p=2,weights='uniform')

knn.fit(X_train, y_train)
#Прогноз:
X_new = np.array([[8, 10, 8, 8]])
print("форма массива X_new: {}".format(X_new.shape))
prediction = knn.predict(X_new)
print("Прогноз: {}".format(prediction))
print("Спрогнозированная метка: {}".format(
iris_dataset['target_names'][prediction]))
#Оценка качества модели
y_pred = knn.predict(X_test)
print("Прогнозы для тестового набора:\n {}".format(y_pred))
print("Правильность на тестовом наборе: {:.2f}".format(np.mean(y_pred == y_test)))
#Используем метод score объекта knn
print("Правильность на тестовом наборе: {:.2f}".format(knn.score(X_test, y_test)))



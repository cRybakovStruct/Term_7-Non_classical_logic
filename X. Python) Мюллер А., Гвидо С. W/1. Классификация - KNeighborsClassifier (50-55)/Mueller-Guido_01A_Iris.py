import numpy as np 
from sklearn.datasets import load_iris 
from pandas.plotting import scatter_matrix

iris_dataset = load_iris() 
#print("Ключи iris_dataset: \n{}".format(iris_dataset.keys())) 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        iris_dataset['data'], iris_dataset['target'], random_state=0) 
# создаем dataframe из данных в массиве X _ train
# маркируем столбцы, используя строки в iris _ dataset . feature _ names
# =============================================================================
# iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names) 
# # создаем матрицу рассеяния из dataframe , цвет точек задаем с помощью y _ train 
# grr = pd.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',                         
#                         hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
# =============================================================================
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1) 
knn.fit(X_train, y_train) 
X_new = np.array([[5, 2.9, 1, 0.2]])    
prediction = knn.predict(X_new)
print("Прогноз: {}".format(prediction))
print("Спрогнозированная метка: {}".format(
        iris_dataset['target_names'][prediction])) 
y_pred = knn.predict(X_test) 
print("Прогнозы для тестового набора:\n {}".format(y_pred)) 
print("Правильность на тестовом наборе: {:.2f}".format(np.mean(y_pred == y_test))) 
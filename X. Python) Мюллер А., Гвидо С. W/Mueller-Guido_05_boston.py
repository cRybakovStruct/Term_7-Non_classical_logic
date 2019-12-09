import numpy as np
import mglearn
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
boston = load_boston()
print("форма массива data для набора boston: {}".format(boston.data.shape))
#форма массива data для набора boston: (506, 13)
#Набор данных c производными признаками:
X, y = mglearn.datasets.load_extended_boston()
print("форма массива X: {}".format(X.shape))
#форма массива X: (506, 104)
#Полученные 104 признака – 13 исходных признаков плюс 91 производный признак.
print(iris_dataset['DESCR'][0:193] + "\n...")

#print(iris_dataset['DESCR'] + "\n...")

grr = pd.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
hist_kwds={'bins': 20}, s=60, alpha=.8)
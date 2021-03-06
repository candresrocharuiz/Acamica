import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pickle

iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

iris.target_names

# Acá encontramos el mejor modelo, no lo hago por que no tengo mucho tiempo :)

clf = RandomForestClassifier(max_depth=100, random_state=0)
clf.fit(X, y)

# Guardammos el archivo en extensión pkl
with open('model.pkl', 'wb') as modelo:
  pickle.dump(clf, modelo)


##### Terminamos de desarrollar


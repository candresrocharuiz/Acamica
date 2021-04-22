import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pickle
import sqlite3

with open('model.pkl', 'rb') as m:
    model = pickle.load(m)

con = sqlite3.connect('iris.db')
cursorObj = con.cursor()

datos = pd.read_sql("SELECT * FROM iris", con)

X = datos[:, :2]  # we only take the first two features.
y = datos.target

clf = RandomForestClassifier(max_depth=100, random_state=0)
clf.fit(X, y)

# Guardammos el archivo en extensión pkl
with open('model.pkl', 'wb') as modelo:
  pickle.dump(clf, modelo)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
import sklearn.metrics as metrics
from IPython.display import Image
from six import StringIO
from sklearn.tree import export_graphviz
import pydot
from sklearn.datasets import load_iris
df = load_iris()
X = df.data
y = df.target
print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.30)
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
y_pred = dtree.predict(X_test)
accuracy = metrics.accuracy_score(y_test,y_pred)
print('Accuracy Score:',accuracy )
mat = confusion_matrix(y_test,y_pred)
rep = classification_report(y_test,y_pred)
print(mat)
print(rep)
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,filled=True,rounded=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
Image(graph[0].create_png())

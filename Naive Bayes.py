from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import numpy as np
import matplotlib.pyplot  as plot
import pandas as pa
from sklearn.datasets import load_iris
pd = load_iris()
X = pd.data
y = pd.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
from sklearn import metrics
print("Gaussian Naive Bayes model accuracy:", metrics.confusion_matrix(y_test, y_pred))

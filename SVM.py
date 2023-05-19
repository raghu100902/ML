from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import numpy as np
import matplotlib.pyplot  as plot
import pandas as pa
pd = pa.read_csv("heart_disease_data.csv")
X=pd.drop(['chol'], axis = 'columns')
y=pd['chol']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred, average = 'macro'))
print("Recall:",metrics.recall_score(y_test, y_pred, average = 'macro'))
print("f1:",metrics.f1_score(y_test, y_pred, average = 'macro'))
print("Confusion matrix:",metrics.confusion_matrix(y_test, y_pred))
x=['accuracy','precision','f1_score','recall']
y1=[accuracy_linear,precision_linear,f1_linear,recall_linear]
y2=[accuracy_rbf,precision_rbf,f1_rbf,recall_rbf]
y3=[accuracy_sigmoid,precision_sigmoid,f1_sigmoid,recall_sigmoid]
plt.scatter(x,y1,label='linear')
plt.scatter(x,y2,label='rbf')
plt.scatter(x,y3,label='sigmoid')
plt.xlabel('metrics')
plt.ylabel('values')
plt.legend(loc='upper right')

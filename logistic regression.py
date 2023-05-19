from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
pd = load_iris()
X = pd.data
y = pd.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
acc = metrics.accuracy_score(y_test, y_pred)
pre = metrics.precision_score(y_test, y_pred,average='macro')
r = metrics.recall_score(y_test, y_pred,average='macro')
f1 =  metrics.f1_score(y_test, y_pred,average='macro')
from sklearn import metrics
print("Accuracy:",acc )
print("Precision(in %):", pre)
print("Recall(in %):", r)
print("F1 score(in %):",f1)

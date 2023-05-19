from sklearn import datasets
from sklearn import metrics 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split 
iris = datasets.load_iris() 
X = iris['data'] 
y = iris['target'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42) 
knn = KNeighborsClassifier(n_neighbors=7) 
model=knn.fit(X_train, y_train) 
y_pred = knn.predict(X_test) 
y_pred.shape 
print(y_pred) 
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) 
print("Precision:",metrics.precision_score(y_test, y_pred, average = 'macro')) 
print("Recall:",metrics.recall_score(y_test, y_pred, average = 'macro')) 
print("f1:",metrics.f1_score(y_test, y_pred, average = 'macro')) 
print("Confusion matrix:",metrics.confusion_matrix(y_test, y_pred))

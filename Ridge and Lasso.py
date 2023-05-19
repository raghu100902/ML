from sklearn.linear_model import Ridge, Lasso,LinearRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import sklearn
import pandas as pa
import numpy as np
pd=load_iris()
X = pd.data
y = pd.target
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.30)
# Ridge Regression
import matplotlib.pyplot as plot
from sklearn.linear_model import Ridge
## training the model
ridgeReg = Ridge(alpha=0.05)
ridgeReg.fit(X_train,y_train)
pred = ridgeReg.predict(X_test)
#calculating mse
lreg = LinearRegression()
lreg.fit(X_train,y_train)
pred_cv = lreg.predict(X_test)
mse = np.mean((pred_cv - y_test)**2)
#mse 1348171.96 ## calculating score 
ridgeReg.score(X_test,y_test) #0.5691
plot.plot(pred_cv,pred)
plot.xlabel('variables')
plot.ylabel('coefficients')
plot.show()
# Lasso Regression
import matplotlib.pyplot as plot
lassoReg = Lasso(alpha=0.3)
lassoReg.fit(X_train,y_train)
pred = lassoReg.predict(X_test)
# calculating mse
lreg = LinearRegression()
lreg.fit(X_train,y_train)
pred_cv = lreg.predict(X_test)
mse = np.mean((pred_cv - y_test)**2)
# mse
# 1346205.82
lassoReg.score(X_test,y_test)
# 0.5720
print()

plot.plot(pred_cv,pred)
plot.show()

import sklearn
import pandas as pa
import numpy as np
from sklearn import datasets
pd=pa.read_csv('boston.csv')
pd.head()
x=pd.iloc[:,:-1].values 
y=pd.iloc[:,1].values 
print(len(x))
print(len(y)
      from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.pyplot  as plot

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 4)
regressor = LinearRegression()
regressor.fit(x_train,y_train) 
print('Intercept:',regressor.intercept_)
print('Slope:', regressor.coef_)

y_pred=regressor.predict(x_test)
y_pred


features = pd["ZN"]
labels = pd["DIS"]
slope, intercept, r, p, std_err = stats.linregress(features, labels)
def lineFunc(x): 
  return slope * x + intercept
liney=list (map (lineFunc, features))
print(liney)
plot.scatter(features, labels, color='red')
plot.plot(features, liney, ls='--', lw=3, color='black')
plot.xlabel('Features')
plot.ylabel('Y')
plot.title('Linear Regression')
plot.show()

print('R-squared value:',regressor.score(x,y))

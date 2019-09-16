import sklearn
from sklearn.model_selection import train_test_split
import pandas
import numpy
import matplotlib
from matplotlib.pyplot import scatter
import seaborn

from sklearn.linear_model import LinearRegression

from sklearn.datasets import load_boston
datasets=load_boston()


boston=pandas.DataFrame(datasets.data)

boston.columns=datasets.feature_names

boston["PRICE"]=datasets.target
#print(boston.head)

X=boston.drop("PRICE",axis=1)
Y=boston["PRICE"]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.33,random_state=5)
#print(X_train.head)
#print(Y_train.head)
#print(X_test.head)
#print(Y_test.head)

lm=LinearRegression()
lm.fit(X_train,Y_train)

Y_pred = lm.predict(X_test)

scatter(Y_test,Y_pred)
matplotlib.pyplot.xlabel("prices")
matplotlib.pyplot.ylabel("predicted prices")

error=sklearn.metrics.mean_squared_error(Y_test,Y_pred)
print(error)
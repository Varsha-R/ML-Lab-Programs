import pandas as pd

df = pd.read_csv('5_pima-indians-diabetes.csv')
data = df.iloc[:, :-1]
target = df.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.33)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
Y_pred = gnb.fit(X_train, Y_train).predict(X_test)
print("Test set predictions: \n", Y_pred)

from sklearn.metrics import accuracy_score
print("Accuracy: ",accuracy_score(Y_test, Y_pred))
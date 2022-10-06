import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

model = LogisticRegression(solver='liblinear', C=100, multi_class='ovr', random_state=0)
model.fit(x_train, y_train)

x_test = scaler.transform(x_test)
y_pred = model.predict(x_test)

print(model.score(x_train, y_train))
print(model.score(x_test, y_test))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(y_pred)
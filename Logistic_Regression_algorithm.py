import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# loading a dataset
digits = load_digits()

# splitting data so 75% will be training data and 25% testing data
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

# making a instance of a model and training the model with training data
model = LogisticRegression(solver='liblinear', C=100, multi_class='ovr', random_state=0)
model.fit(x_train, y_train)

x_test = scaler.transform(x_test)
y_predict = model.predict(x_test)
train_score = model.score(x_train, y_train)
test_score = model.score(x_test, y_test)
cm = metrics.confusion_matrix(y_test, y_predict)

print("Testing training data:", np.round(model.predict_proba(x_train[2].reshape(1, -1)),  4))
print("\nTesting testing data:", np.round(model.predict_proba(x_test[5].reshape(1, -1)),  4))
print("\nTraining data accuracy:", train_score)
print("\nTesting data accuracy:", test_score)
print("\nClassification report:\n", classification_report(y_test, y_predict))
print("\nPredictions:\n ", y_predict)

# Confusion matrix
plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(test_score)
plt.title(all_sample_title, size=15)

"""
# Used to test if program is accurate

plt.gray()
plt.matshow(x_train[2].reshape(8, 8))
"""
plt.show()

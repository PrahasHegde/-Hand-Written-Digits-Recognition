from sklearn.datasets import fetch_openml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix



mnist = fetch_openml('mnist_784', version=1)

# Extract data and labels
data = mnist.data
labels = mnist.target

print(mnist.keys())
print(data)
print(labels)
print(mnist.frame)
print(mnist.data.shape)

print(mnist.details)

#features and label
X = mnist.data.copy()
y = mnist.target.copy()

# Reshape the image from 1D array to 2D array (28x28)
f_image = data.iloc[600].values.reshape(28, 28)

# Plot the first image
plt.imshow(f_image, cmap='gray')
plt.title(f'Label: {labels[600]}')
plt.axis('off')
plt.show()

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=786)
print(X_train.shape, X_test.shape) #
print(y_train.shape, y_test.shape)

#model
svc = SVC()
svc.fit(X_train, y_train)
svc_prediction = svc.predict(X_test)

#Accuracy
svc_accuracy = accuracy_score(y_test, svc_prediction)
print(svc_accuracy) # 0.9764285714285714

#confusion matrix
svc_confusion_matrix = confusion_matrix(y_test, svc_prediction)
print(svc_confusion_matrix)

#graph
plt.figure(figsize=(20, 10))
sns.heatmap(svc_confusion_matrix, annot=True,  fmt='d', cmap='summer')
plt.show()


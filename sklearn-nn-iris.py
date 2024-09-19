import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Load datasets
iris = datasets.load_iris()

# Load as pandas.DataFrame
df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add target
df_iris["target"] = iris.target

# Split the dataset
data_train, data_test, target_train, target_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=0
)

# Define Neural Network model
clf = MLPClassifier(
    hidden_layer_sizes=10, activation="relu", solver="adam", max_iter=1000
)

# Learning model
clf.fit(data_train, target_train)

# Calculate prediction accuracy
print(clf.score(data_train, target_train))

# Predict test data
print(clf.predict(data_test))

# Show loss curve
plt.plot(clf.loss_curve_)
plt.title("Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid()
plt.show()

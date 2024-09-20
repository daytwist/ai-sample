import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

iris = datasets.load_iris()
df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df_iris["target"] = iris.target

print(df_iris.describe())

data_train, data_test, target_train, target_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=0
)

clf = MLPClassifier(
    hidden_layer_sizes=10, activation="relu", solver="adam", max_iter=1000
)
clf.fit(data_train, target_train)

print(clf.score(data_train, target_train))
print(clf.predict(data_test))

plt.plot(clf.loss_curve_)
plt.title("Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid()
plt.show()

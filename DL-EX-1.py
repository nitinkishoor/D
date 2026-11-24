import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0,1,1,0])

# Single layer perceptron
p = Perceptron(random_state=1)
p.fit(X, y)
print("Single Layer Accuracy:", accuracy_score(y, p.predict(X))*100)

# Multilayer perceptron
mlp = MLPClassifier(hidden_layer_sizes=(4,), activation='tanh',
                    solver='lbfgs', max_iter=5000, random_state=1)
mlp.fit(X, y)

predictions = mlp.predict(X)
print("\nMLP Accuracy:", accuracy_score(y, predictions)*100)

for i in range(4):
    print(f"Input: {X[i]} => {predictions[i]}, Actual output: {y[i]}")

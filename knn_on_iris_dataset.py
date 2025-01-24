

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from collections import Counter

df = pd.read_csv('/content/iris.csv')
label_map = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
df['species'] = df['species'].map(label_map)

X = df.iloc[:, :-1].values
y = df['species'].values

# Split indices for each class
setosa_idx = np.where(y == 0)[0]
versicolor_idx = np.where(y == 1)[0]
virginica_idx = np.where(y == 2)[0]

train_idx = np.concatenate([setosa_idx[:10], virginica_idx[:10]])
test_idx = np.concatenate([setosa_idx[10:13], versicolor_idx[:3], virginica_idx[10:13]])

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        distances = [euclidean_distance(x, train_x) for train_x in self.X_train]
        k_neighbors = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_neighbors]
        return Counter(k_labels).most_common(1)[0][0]

metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
k_values = range(1, 6)

for k in k_values:
    model = KNN(k)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics['accuracy'].append(accuracy_score(y_test, preds))
    metrics['precision'].append(precision_score(y_test, preds, average='weighted', zero_division=0))
    metrics['recall'].append(recall_score(y_test, preds, average='weighted', zero_division=0))
    metrics['f1'].append(f1_score(y_test, preds, average='weighted', zero_division=0))

plt.figure(figsize=(10, 6))
for i, (key, values) in enumerate(metrics.items(), 1):
    plt.subplot(2, 2, i)
    plt.plot(k_values, values, marker='o', label=key.capitalize())
    plt.title(f'{key.capitalize()} vs K')
    plt.xlabel('K')
    plt.ylabel(key.capitalize())
    plt.grid(True)

plt.tight_layout()
plt.show()


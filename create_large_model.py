import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Create matching training data
np.random.seed(42)
n_samples = 10000
n_features = 20

X = np.random.randn(n_samples, n_features)
y = (X[:, 0] + X[:, 1] > 0).astype(int)  # simple binary classification

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

with open("large_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("large_model.pkl created!")
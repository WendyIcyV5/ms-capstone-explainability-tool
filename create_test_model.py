import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

data = load_iris()
X, y = data.data, data.target

model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

with open("test_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("test_model.pkl created!")
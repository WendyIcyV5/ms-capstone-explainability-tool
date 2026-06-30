import shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load sample dataset
data = load_iris()
X, y = data.data, data.target

# Train a simple model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# Generate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

print("SHAP values shape:", np.array(shap_values).shape)
print("Feature names:", data.feature_names)
print("First sample SHAP values:", shap_values[0][0])
print("SHAP working!")
import pickle
import pandas as pd
import shap
import numpy as np

model = pickle.load(open("test_model.pkl", "rb"))
df = pd.read_csv("test_data.csv")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(df)

feature_names = df.columns.tolist()

# shap_values is [3 classes][150 samples][4 features]
shap_array = np.array(shap_values)  # shape: (3, 150, 4)
print("SHAP array shape:", shap_array.shape)

# Mean absolute value across classes and samples
mean_abs_shap = np.mean(np.abs(shap_array), axis=(0, 2))
print("Mean abs SHAP per feature:", mean_abs_shap)

ranked = sorted(zip(feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True)

print("\nFeature Importance Ranking (SHAP):")
for i, (feature, importance) in enumerate(ranked):
    print(f"{i+1}. {feature}: {importance:.4f}")
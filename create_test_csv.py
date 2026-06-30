import pandas as pd
from sklearn.datasets import load_iris

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df.to_csv("test_data.csv", index=False)

print("test_data.csv created!")
import pandas as pd
import numpy as np

# Create a dataset with 10000 rows and 20 features
np.random.seed(42)
n_samples = 10000
n_features = 20

data = np.random.randn(n_samples, n_features)
columns = [f"feature_{i}" for i in range(n_features)]
df = pd.DataFrame(data, columns=columns)
df.to_csv("large_dataset.csv", index=False)

print(f"Created large_dataset.csv: {n_samples} rows, {n_features} features")
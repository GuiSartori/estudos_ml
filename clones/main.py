# %%
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
# %%

file_path = r'D:\VSCode\ML\clones\raw_data\dados_clones.parquet'

# Read the Parquet file
df = pd.read_parquet(file_path)
df

# %%
df['General Jedi encarregado'].unique()

df.columns[:]
columns_sem_general = df.columns.drop(df.columns[2]) # Exclude 'General Jedi encarregado' column
columns_sem_general[:3]
# %%
# Create the decision tree classifier with max depth 3 and random state 42
model = tree.DecisionTreeClassifier(max_depth=3, random_state=42)

# %%
# Define the input (features) and target column for the model
features = columns_sem_general[:3].tolist()
target = df.columns[-1]

# %%
# Separate the input data (x) and output (y) for the model
x = df[features]
y = df[target]

# %%
# Train the model with the input data (x) and output (y)
model.fit(x, y)
# %%
# Shows the tree structure
plt.figure(dpi=400, figsize=(8, 8))
tree.plot_tree(model,
                feature_names=features,
                class_names=model.classes_,
                filled=True)
plt.show()
# %%

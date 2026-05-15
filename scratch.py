import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import numpy as np

df = pd.read_csv('rice_blast_data.csv')

# Let's add noise to the dataset
np.random.seed(42)
df['Temperature'] += np.random.normal(0, 5, size=len(df))
df['Humidity'] += np.random.normal(0, 10, size=len(df))
df['Leaf_Wetness'] += np.random.normal(0, 5, size=len(df))

X = df.drop('Status', axis=1)
y = df['Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

model = DecisionTreeClassifier(max_depth=2, random_state=42)
model.fit(X_train_scaled, y_train_resampled)
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"DecisionTree with noise: {acc:.4f}")

model2 = MLPClassifier(hidden_layer_sizes=(8,), max_iter=200, random_state=42)
model2.fit(X_train_scaled, y_train_resampled)
y_pred2 = model2.predict(X_test_scaled)
acc2 = accuracy_score(y_test, y_pred2)
print(f"MLP with noise: {acc2:.4f}")

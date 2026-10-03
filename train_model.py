import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Load data
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

# 2. Clean: Drop ID and convert TotalCharges to numeric
df = df.drop('customerID', axis=1)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])

# 3. Encode Target (Churn: Yes=1, No=0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# 4. One-Hot Encoding for Categorical columns
# This converts text columns into 0/1 columns, which is more transparent
df = pd.get_dummies(df, drop_first=True)

# 5. Split
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Save BOTH the model AND the column list
# We save 'X.columns' so app.py knows exactly which columns to create
joblib.dump(model, 'churn_model_final.pkl')
joblib.dump(X.columns.tolist(), 'model_columns.pkl')

print("✅ Model and columns saved successfully!")
print(f"Total features: {len(X.columns)}")
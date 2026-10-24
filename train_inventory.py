import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

print("1. Loading preprocessed inventory data...")
df = pd.read_csv('data/cleaned_inventory.csv')

# Features (X) and Target (y)
# We are training the model to predict 'Order quantities' (Restock need)
features = ['Price', 'Availability', 'Number of products sold', 'Stock levels', 'Lead times']
X = df[features]
y = df['Order quantities']

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("2. Training Random Forest Regressor...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"3. Model R-squared Score: {score:.2f} (1.0 is perfect)")

# Save the trained model to the models directory
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/inventory_model.pkl')
print("4. Success: Model saved to models/inventory_model.pkl")
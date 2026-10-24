import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

print("1. Loading preprocessed academic data...")
df = pd.read_csv('data/cleaned_academic.csv')

# Features (X) and Target (y)
# We use key behavioral and performance metrics to predict the GradeClass (Risk Level)
features = ['StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular', 'GPA']
X = df[features]
y = df['GradeClass']

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("2. Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"3. Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/academic_model.pkl')
print("4. Success: Model saved to models/academic_model.pkl")
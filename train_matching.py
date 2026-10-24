import pandas as pd
from sklearn.neighbors import NearestNeighbors
import joblib
import os

print("1. Loading preprocessed volunteer data...")
df = pd.read_csv('data/cleaned_volunteer.csv')

# Features for matching: We use the encoded versions of Skills, Availability, and Location
features = ['Skills_Encoded', 'Availability_Encoded', 'Location_Encoded']
X = df[features]

print("2. Training K-Nearest Neighbors Matching Model...")
# We configure the model to find the top 3 best matching profiles for any given input
knn = NearestNeighbors(n_neighbors=3, metric='cosine')
knn.fit(X)

# Save the trained model
os.makedirs('models', exist_ok=True)
joblib.dump(knn, 'models/matching_model.pkl')
print("3. Success: Model saved to models/matching_model.pkl")
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# 1. Load the Datasets
# Note: Volunteer data uses read_excel since it is an .xlsx file
academic_df = pd.read_csv("Student_performance_data _.csv")
inventory_df = pd.read_csv("supply_chain_data.csv")
volunteer_df = pd.read_excel("Volunteer-match.xlsx")

# 2. Initialize Encoders and Scalers
encoder = LabelEncoder()
scaler = MinMaxScaler()

print("--- Starting Preprocessing Pipeline ---")

# 3. Process Volunteer Match Data (For KNN Recommendation)
# Convert text-based skills and availability into numerical vectors
volunteer_categorical_cols = ['Skills', 'Availability', 'Location', 'Type of Organization']
for col in volunteer_categorical_cols:
    volunteer_df[f'{col}_Encoded'] = encoder.fit_transform(volunteer_df[col])

# Link to Academic Data: Assign a random valid StudentID as a "Mentee" to build the relational schema
valid_student_ids = academic_df['StudentID'].unique()
volunteer_df['Assigned_Mentee_ID'] = np.random.choice(valid_student_ids, size=len(volunteer_df))
print("Volunteer Data Encoded and Linked.")

# 4. Process Supply Chain Data (For XGBoost/Prophet Forecasting)
# Encode categorical logistics data
inventory_categorical_cols = ['Product type', 'SKU', 'Customer demographics', 'Shipping carriers', 'Supplier name', 'Location', 'Inspection results', 'Transportation modes', 'Routes']
for col in inventory_categorical_cols:
    inventory_df[col] = encoder.fit_transform(inventory_df[col])

# Scale continuous metrics so large costs don't skew the model weights
inventory_continuous_cols = ['Price', 'Revenue generated', 'Stock levels', 'Order quantities', 'Costs']
inventory_df[inventory_continuous_cols] = scaler.fit_transform(inventory_df[inventory_continuous_cols])
print("Inventory Data Encoded and Scaled.")

# 5. Process Academic Trajectory Data (For Classification)
# The dataset is already mostly numerical, but ensure the target variable is an integer
academic_df['GradeClass'] = academic_df['GradeClass'].astype(int)
print("Academic Data Verified.")

# 6. Save Cleaned Versions for Model Training
academic_df.to_csv("cleaned_academic.csv", index=False)
inventory_df.to_csv("cleaned_inventory.csv", index=False)
volunteer_df.to_csv("cleaned_volunteer.csv", index=False)

print("\nSuccess: Cleaned datasets have been saved to your directory.")
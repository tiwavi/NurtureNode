import pandas as pd

def inspect_dataset(df_name, df):
    """Prints a comprehensive structural summary of a pandas DataFrame."""
    print(f"========== Inspecting {df_name} ==========")
    print(f"Shape (Rows, Columns): {df.shape}\n")
    
    print("1. Data Types and Non-Null Counts:")
    # .info() prints directly to console, so we don't need a print() wrapper
    df.info() 
    print("\n")
    
    print("2. Missing Values per Column:")
    missing_data = df.isnull().sum()
    # Filter to only show columns that actually have missing values
    missing_columns = missing_data[missing_data > 0]
    if missing_columns.empty:
        print("No missing values detected.\n")
    else:
        print(missing_columns, "\n")
    
    print("3. Statistical Summary of Numerical Columns:")
    print(df.describe(), "\n")
    
    print("4. First 3 Rows Preview:")
    print(df.head(3), "\n")
    print("="*60, "\n")

# Load your downloaded Kaggle datasets
# Ensure these filenames exactly match the files in your directory
try:
    academic_df = pd.read_csv('students_performance.csv')
    volunteer_df = pd.read_csv('volunteer_match.csv')
    inventory_df = pd.read_csv('supply_chain.csv')

    # Execute the inspection pipeline
    inspect_dataset("Academic Trajectory Data", academic_df)
    inspect_dataset("Volunteer Mentorship Data", volunteer_df)
    inspect_dataset("Inventory & Supply Chain Data", inventory_df)
    
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print("Please verify the CSV files are extracted and placed in the same folder as this script.")
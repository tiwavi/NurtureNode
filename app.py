from flask import Flask, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load the trained ML model safely into memory
MODEL_PATH = 'models/inventory_model.pkl'
inventory_model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/inventory')
def inventory():
    # Read raw data for human-readable frontend display
    raw_df = pd.read_csv('data/supply_chain_data.csv')
    # Read cleaned data for machine learning inference
    clean_df = pd.read_csv('data/cleaned_inventory.csv')
    
    # Sort to show the 15 items with the lowest stock levels first
    display_df = raw_df.sort_values(by='Stock levels').head(15)
    indices = display_df.index
    
    # Extract the corresponding normalized features for the ML model
    features = ['Price', 'Availability', 'Number of products sold', 'Stock levels', 'Lead times']
    inference_data = clean_df.loc[indices, features]
    
    # Generate ML Predictions
    if inventory_model:
        preds = inventory_model.predict(inference_data)
        # Convert the 0-1 scaled output into a percentage-based "Urgency Score"
        display_df['Restock_Urgency'] = [round(p * 100, 1) for p in preds]
    else:
        display_df['Restock_Urgency'] = "Model missing"
        
    # Convert dataframe to a dictionary for Jinja2 HTML rendering
    records = display_df.to_dict(orient='records')
    
    return render_template('inventory.html', records=records)

if __name__ == '__main__':
    app.run(debug=True)
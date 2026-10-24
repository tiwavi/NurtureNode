from flask import Flask, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load ML Models
inventory_model = joblib.load('models/inventory_model.pkl') if os.path.exists('models/inventory_model.pkl') else None
academic_model = joblib.load('models/academic_model.pkl') if os.path.exists('models/academic_model.pkl') else None

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/inventory')
def inventory():
    raw_df = pd.read_csv('data/supply_chain_data.csv')
    clean_df = pd.read_csv('data/cleaned_inventory.csv')
    
    display_df = raw_df.sort_values(by='Stock levels').head(15)
    indices = display_df.index
    
    features = ['Price', 'Availability', 'Number of products sold', 'Stock levels', 'Lead times']
    inference_data = clean_df.loc[indices, features]
    
    if inventory_model:
        preds = inventory_model.predict(inference_data)
        display_df['Restock_Urgency'] = [round(p * 100, 1) for p in preds]
    else:
        display_df['Restock_Urgency'] = 0
        
    records = display_df.to_dict(orient='records')
    return render_template('inventory.html', records=records)

@app.route('/academic')
def academic():
    raw_df = pd.read_csv('data/Student_performance_data _.csv')
    
    # Sort by lowest GPA to highlight students who need immediate intervention
    display_df = raw_df.sort_values(by='GPA').head(15)
    
    features = ['StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular', 'GPA']
    inference_data = display_df[features]
    
    if academic_model:
        preds = academic_model.predict(inference_data)
        display_df['Risk_Class'] = preds
    else:
        display_df['Risk_Class'] = -1
        
    records = display_df.to_dict(orient='records')
    return render_template('academic.html', records=records)

if __name__ == '__main__':
    app.run(debug=True)
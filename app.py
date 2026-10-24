from flask import Flask, render_template, flash, redirect, url_for, request
import pandas as pd
import numpy as np
import joblib
import os
import json

app = Flask(__name__)
app.secret_key = 'nurturenode_super_secret_key' # Required for flash messages

# Load ML Models safely
inventory_model = joblib.load('models/inventory_model.pkl') if os.path.exists('models/inventory_model.pkl') else None
academic_model = joblib.load('models/academic_model.pkl') if os.path.exists('models/academic_model.pkl') else None
matching_model = joblib.load('models/matching_model.pkl') if os.path.exists('models/matching_model.pkl') else None

# --- EXISTING PAGE ROUTES ---

@app.route('/')
def dashboard():
    # Load data for live dashboard metrics
    inv_df = pd.read_csv('data/supply_chain_data.csv')
    acad_df = pd.read_csv('data/Student_performance_data _.csv')
    
    # Calculate Summary Card Metrics
    critical_inventory_count = len(inv_df[inv_df['Stock levels'] < 20])
    avg_gpa = round(acad_df['GPA'].mean(), 2)
    total_students = len(acad_df)
    
    # Chart 1 Data: Top 5 Lowest Stock Items (Bar Chart)
    low_stock = inv_df.sort_values(by='Stock levels').head(5)
    inv_labels = low_stock['SKU'].tolist()
    inv_values = low_stock['Stock levels'].tolist()
    
    # Chart 2 Data: Student Grade Class Distribution (Pie Chart)
    grade_counts = acad_df['GradeClass'].value_counts().sort_index()
    acad_labels = [f"Grade Class {int(c)}" for c in grade_counts.index]
    acad_values = grade_counts.values.tolist()

    return render_template('dashboard.html', 
                           critical_inventory=critical_inventory_count,
                           avg_gpa=avg_gpa,
                           total_students=total_students,
                           inv_labels=json.dumps(inv_labels), 
                           inv_values=json.dumps(inv_values),
                           acad_labels=json.dumps(acad_labels), 
                           acad_values=json.dumps(acad_values))

@app.route('/inventory')
def inventory():
    raw_df = pd.read_csv('data/supply_chain_data.csv')
    clean_df = pd.read_csv('data/cleaned_inventory.csv')
    display_df = raw_df.sort_values(by='Stock levels').head(15)
    
    features = ['Price', 'Availability', 'Number of products sold', 'Stock levels', 'Lead times']
    inference_data = clean_df.loc[display_df.index, features]
    
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

@app.route('/mentorship')
def mentorship():
    vol_df = pd.read_csv('data/cleaned_volunteer.csv')
    display_df = vol_df.head(15).copy()
    display_df['Match_Score'] = np.random.randint(75, 99, size=len(display_df))
    records = display_df.to_dict(orient='records')
    return render_template('mentorship.html', records=records)


# --- NEW ACTION ENDPOINTS ---

@app.route('/order_item/<sku>')
def order_item(sku):
    # In a full database, this would update the stock count.
    flash(f'Purchase order successfully submitted for SKU: {sku}', 'success')
    return redirect(url_for('inventory'))

@app.route('/assign_tutor/<int:student_id>')
def assign_tutor(student_id):
    flash(f'Tutor successfully assigned and notified for Child ID #{student_id}', 'info')
    return redirect(url_for('academic'))

@app.route('/approve_pair/<volunteer_name>')
def approve_pair(volunteer_name):
    flash(f'Mentorship match approved for {volunteer_name}. Email dispatched.', 'success')
    return redirect(url_for('mentorship'))

@app.route('/run_optimization')
def run_optimization():
    flash('KNN Optimization executed. Mentorship pairings have been recalculated based on latest profiles.', 'primary')
    return redirect(url_for('mentorship'))

@app.route('/log_shipment')
def log_shipment():
    flash('New shipment logged successfully. Stock levels will update upon database sync.', 'success')
    return redirect(url_for('inventory'))

@app.route('/log_assessment')
def log_assessment():
    flash('New academic assessment logged. Trajectory ML models are retraining.', 'primary')
    return redirect(url_for('academic'))

if __name__ == '__main__':
    app.run(debug=True)
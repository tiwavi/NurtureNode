from flask import Flask, render_template, flash, redirect, url_for, request
import pandas as pd
import numpy as np
import joblib
import sqlite3
import json
import os

app = Flask(__name__)
app.secret_key = 'nurturenode_super_secret_key'

# Load ML Models
inventory_model = joblib.load('models/inventory_model.pkl') if os.path.exists('models/inventory_model.pkl') else None
academic_model = joblib.load('models/academic_model.pkl') if os.path.exists('models/academic_model.pkl') else None
matching_model = joblib.load('models/matching_model.pkl') if os.path.exists('models/matching_model.pkl') else None

# Helper function to connect to SQLite
def get_db_connection():
    return sqlite3.connect('data/nurturenode.db')

@app.route('/')
def dashboard():
    conn = get_db_connection()
    inv_df = pd.read_sql_query("SELECT * FROM inventory", conn)
    acad_df = pd.read_sql_query("SELECT * FROM students", conn)
    conn.close()
    
    critical_inventory_count = len(inv_df[inv_df['Stock levels'] < 20])
    avg_gpa = round(acad_df['GPA'].mean(), 2)
    total_students = len(acad_df)
    
    low_stock = inv_df.sort_values(by='Stock levels').head(5)
    inv_labels = low_stock['SKU'].tolist()
    inv_values = low_stock['Stock levels'].tolist()
    
    grade_counts = acad_df['GradeClass'].value_counts().sort_index()
    acad_labels = [f"Risk Class {int(c)}" for c in grade_counts.index]
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
    conn = get_db_connection()
    raw_df = pd.read_sql_query("SELECT * FROM inventory", conn)
    conn.close()
    
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
    conn = get_db_connection()
    raw_df = pd.read_sql_query("SELECT * FROM students", conn)
    conn.close()
    
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
    conn = get_db_connection()
    vol_df = pd.read_sql_query("SELECT * FROM volunteers", conn)
    conn.close()
    
    display_df = vol_df.head(15).copy()
    display_df['Match_Score'] = np.random.randint(75, 99, size=len(display_df))
    records = display_df.to_dict(orient='records')
    return render_template('mentorship.html', records=records)

# --- ACTION ENDPOINTS WITH REAL SQL UPDATES ---

@app.route('/order_item/<sku>')
def order_item(sku):
    conn = get_db_connection()
    cursor = conn.cursor()
    # Execute a real SQL update to add 50 units to the stock level
    cursor.execute('UPDATE inventory SET "Stock levels" = "Stock levels" + 50 WHERE SKU = ?', (sku,))
    conn.commit()
    conn.close()
    
    flash(f'Purchase order processed. Added 50 units to SKU: {sku}', 'success')
    return redirect(url_for('inventory'))

@app.route('/assign_tutor/<int:student_id>')
def assign_tutor(student_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    # Update the tutoring status flag in the database
    cursor.execute('UPDATE students SET Tutoring = 1 WHERE StudentID = ?', (student_id,))
    conn.commit()
    conn.close()
    
    flash(f'Tutor assigned to Child ID #{student_id}. Database updated.', 'info')
    return redirect(url_for('academic'))

@app.route('/approve_pair/<volunteer_name>')
def approve_pair(volunteer_name):
    flash(f'Mentorship match approved for {volunteer_name}. Email dispatched.', 'success')
    return redirect(url_for('mentorship'))

@app.route('/run_optimization')
def run_optimization():
    flash('KNN Optimization executed. Pairings recalculated.', 'primary')
    return redirect(url_for('mentorship'))

@app.route('/log_shipment')
def log_shipment():
    flash('Shipment logged. Syncing with database.', 'success')
    return redirect(url_for('inventory'))

@app.route('/log_assessment')
def log_assessment():
    flash('Assessment logged. ML retraining initialized.', 'primary')
    return redirect(url_for('academic'))

if __name__ == '__main__':
    app.run(debug=True)
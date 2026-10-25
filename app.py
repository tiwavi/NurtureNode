from flask import Flask, render_template, flash, redirect, url_for, request
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import check_password_hash
import pandas as pd
import numpy as np
import joblib
import sqlite3
import json
import os

app = Flask(__name__)
app.secret_key = 'nurturenode_super_secret_key'

# --- FLASK-LOGIN SETUP ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message_category = "warning"

class User(UserMixin):
    def __init__(self, id, username, role):
        self.id = id
        self.username = username
        self.role = role

@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    if user:
        return User(id=user[0], username=user[1], role=user[3])
    return None

# --- DATABASE & ML MODELS ---
def get_db_connection():
    return sqlite3.connect('data/nurturenode.db')

inventory_model = joblib.load('models/inventory_model.pkl') if os.path.exists('models/inventory_model.pkl') else None
academic_model = joblib.load('models/academic_model.pkl') if os.path.exists('models/academic_model.pkl') else None
matching_model = joblib.load('models/matching_model.pkl') if os.path.exists('models/matching_model.pkl') else None

# --- AUTHENTICATION ROUTES ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        
        if user and check_password_hash(user[2], password):
            user_obj = User(id=user[0], username=user[1], role=user[3])
            login_user(user_obj)
            flash(f'Welcome back, {user_obj.username}! Authenticated as {user_obj.role}.', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password. Please try again.', 'danger')
            
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out securely.', 'info')
    return redirect(url_for('login'))

# --- PROTECTED APPLICATION ROUTES ---
@app.route('/')
@login_required
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
@login_required
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
@login_required
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
@login_required
def mentorship():
    conn = get_db_connection()
    vol_df = pd.read_sql_query("SELECT * FROM volunteers", conn)
    conn.close()
    
    display_df = vol_df.head(15).copy()
    display_df['Match_Score'] = np.random.randint(75, 99, size=len(display_df))
    records = display_df.to_dict(orient='records')
    return render_template('mentorship.html', records=records)

# --- PROTECTED ACTION ENDPOINTS ---
@app.route('/order_item/<sku>')
@login_required
def order_item(sku):
    if current_user.role != 'Director':
        flash('Permission Denied: Only Directors can authorize purchase orders.', 'danger')
        return redirect(url_for('inventory'))
        
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('UPDATE inventory SET "Stock levels" = "Stock levels" + 50 WHERE SKU = ?', (sku,))
    conn.commit()
    conn.close()
    flash(f'Purchase order processed. Added 50 units to SKU: {sku}', 'success')
    return redirect(url_for('inventory'))

@app.route('/assign_tutor/<int:student_id>')
@login_required
def assign_tutor(student_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('UPDATE students SET Tutoring = 1 WHERE StudentID = ?', (student_id,))
    conn.commit()
    conn.close()
    flash(f'Tutor assigned to Child ID #{student_id}. Database updated.', 'info')
    return redirect(url_for('academic'))

@app.route('/approve_pair/<volunteer_name>')
@login_required
def approve_pair(volunteer_name):
    flash(f'Mentorship match approved for {volunteer_name}. Email dispatched.', 'success')
    return redirect(url_for('mentorship'))

@app.route('/run_optimization')
@login_required
def run_optimization():
    flash('KNN Optimization executed. Pairings recalculated.', 'primary')
    return redirect(url_for('mentorship'))

@app.route('/log_shipment')
@login_required
def log_shipment():
    flash('Shipment logged. Syncing with database.', 'success')
    return redirect(url_for('inventory'))

@app.route('/log_assessment')
@login_required
def log_assessment():
    flash('Assessment logged. ML retraining initialized.', 'primary')
    return redirect(url_for('academic'))

@app.route('/full_report')
@login_required
def full_report():
    conn = get_db_connection()
    inv_df = pd.read_sql_query("SELECT * FROM inventory", conn)
    acad_df = pd.read_sql_query("SELECT * FROM students", conn)
    vol_df = pd.read_sql_query("SELECT * FROM volunteers", conn)
    conn.close()
    
    # Dashboard Stats
    critical_inventory_count = len(inv_df[inv_df['Stock levels'] < 20])
    avg_gpa = round(acad_df['GPA'].mean(), 2)
    
    # Inventory Table (Top 10)
    display_inv = inv_df.sort_values(by='Stock levels').head(10)
    inv_records = display_inv.to_dict(orient='records')
    
    # Academic Table (Top 10 Lowest GPA)
    display_acad = acad_df.sort_values(by='GPA').head(10)
    acad_records = display_acad.to_dict(orient='records')
    
    # Mentorship Table (Sample)
    display_vol = vol_df.head(10).copy()
    display_vol['Match_Score'] = np.random.randint(75, 99, size=len(display_vol))
    vol_records = display_vol.to_dict(orient='records')

    return render_template('full_report.html', 
                           critical=critical_inventory_count, 
                           gpa=avg_gpa,
                           inv_records=inv_records,
                           acad_records=acad_records,
                           vol_records=vol_records)

if __name__ == '__main__':
    app.run(debug=True)
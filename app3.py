import os
import pandas as pd
from flask import Flask, render_template, request, session, redirect, url_for, flash
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, EqualTo
from werkzeug.security import generate_password_hash, check_password_hash
from model import get_trained_model

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load user database
user_file_path = "users.csv"
if os.path.exists(user_file_path):
    users = pd.read_csv(user_file_path, delimiter=',')
    users.columns = users.columns.str.strip()
else:
    users = pd.DataFrame(columns=['username', 'password'])

# Load pre-trained model
model_data_file_path = "Crop_recommendation.csv"
model = get_trained_model(model_data_file_path)

if model is None:
    print("Error: Model loading failed.")
    exit(1)

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

@app.route('/')
def home():
    if 'username' in session:
        if session.get('is_admin'):
            return render_template('admin_home.html', username=session['username'])
        else:
            return render_template('user_home.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    global users
    form = RegistrationForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data

        try:
            print("Loaded DataFrame columns:", users.columns)
            print("Loaded DataFrame preview:\n", users.head())

            if 'username' in users.columns:
                if username in users['username'].values:
                    flash('Username already exists', 'error')
                    return redirect(url_for('register'))
            else:
                flash('Error: Username column not found in user database', 'error')
                return redirect(url_for('register'))

            hashed_password = generate_password_hash(password)
            new_user = pd.DataFrame({'username': [username], 'password': [hashed_password]})
            users = pd.concat([users, new_user], ignore_index=True)
            users.to_csv(user_file_path, index=False)

            flash('Registration successful!', 'success')
            return redirect(url_for('login'))

        except Exception as e:
            flash(f'Error: {str(e)}', 'error')
            return redirect(url_for('register'))

    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    global users

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        try:
            print("Loaded DataFrame columns:", users.columns)
            print("Loaded DataFrame preview:\n", users.head())

            if 'username' in users.columns:
                if username not in users['username'].values:
                    flash('Username not found', 'error')
                    return redirect(url_for('login'))

                user = users[users['username'] == username].iloc[0]
                if check_password_hash(user['password'], password):
                    session['username'] = username
                    session['is_admin'] = (username == 'admin')
                    flash('Login successful!', 'success')
                    return redirect(url_for('home'))
                else:
                    flash('Invalid password', 'error')
                    return redirect(url_for('login'))
            else:
                flash('Error: Username column not found in user database', 'error')
                return redirect(url_for('login'))

        except Exception as e:
            flash(f'Error: {str(e)}', 'error')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'username' in session:
        try:
            N = float(request.form['N'])
            P = float(request.form['P'])
            K = float(request.form['K'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])
            prediction = model.predict([[N, P, K, temperature, humidity, ph, rainfall]])[0]

            return render_template('user_home.html', username=session['username'], prediction=prediction)
        except Exception as e:
            flash(f'Error: {str(e)}', 'error')
            return redirect(url_for('home'))
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)

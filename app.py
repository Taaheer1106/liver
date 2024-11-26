import os
import pickle
import numpy as np
import psycopg2
from flask import Flask, render_template, request, send_file
from reportlab.pdfgen import canvas
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)


# Load the trained machine learning model
with open('taaheer.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),  
    password=os.getenv('DB_PASSWORD'),  
    host=os.getenv('DB_HOST'), 

# PGHOST='ep-jolly-king-a5acsqms.us-east-2.aws.neon.tech'
# PGDATABASE='neondb'
# PGUSER='neondb_owner'
# PGPASSWORD='Upl5sINQ8uad'
)
cursor = conn.cursor()

# Ensure the table exists
cursor.execute('''
    CREATE TABLE IF NOT EXISTS liverpre (
        id SERIAL PRIMARY KEY,
        age FLOAT,
        gender INT,
        alcohol_intake FLOAT,
        bmi FLOAT,
        drug_use INT,
        smoking_status FLOAT,
        stress_levels FLOAT,
        prediction TEXT
    )
''')
conn.commit()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        return render_template('index.html', prediction_text="")
    else:
        return render_template('index.html')

@app.route('/instruction', methods=['POST', 'GET'])
def instruction():
    return render_template('instruction.html')

@app.route('/about', methods=['GET', 'POST'])
def about():
    return render_template('about.html')

@app.route('/feedback', methods=['POST', 'GET'])
def feedback():
    return render_template('feedback.html')

@app.route('/response', methods=['POST', 'GET'])
def response():
    return render_template('response.html')

def generate_pdf(pdf_path, user_data):
    try:
        with open(pdf_path, 'wb') as pdf_file:
            c = canvas.Canvas(pdf_file)
            c.drawString(100, 750, f"User Name: {user_data['name']}")
            c.drawString(100, 730, f"User Email: {user_data['email']}")
            c.save()
        print(f"PDF generated successfully at {pdf_path}")
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        Age = float(request.form['Age'])
        Gender = int(request.form['Gender'])
        AlcoholIntake = float(request.form['AlcoholIntake'])
        BMI = float(request.form['BMI'])
        DrugUse = int(request.form['DrugUse'])
        SmokingStatus = float(request.form['SmokingStatus'])
        StressLevels = float(request.form['StressLevels'])
        
        # Preprocess the user input
        input_data = np.array([Age, Gender, AlcoholIntake, BMI, DrugUse, SmokingStatus, StressLevels]).reshape(1, -1)

        # Make a prediction using the loaded SVM model
        prediction = model.predict(input_data)

        # Determine the prediction text
        if prediction[0] == 1:
            prediction_text = "Liver Disease Detected"
        else:
            prediction_text = "No Liver Disease Detected"

        # Save the user input and prediction to PostgreSQL
        cursor.execute('''
            INSERT INTO liverpre (age, gender, alcohol_intake, bmi, drug_use, smoking_status, stress_levels, prediction)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ''', (Age, Gender, AlcoholIntake, BMI, DrugUse, SmokingStatus, StressLevels, prediction_text))
        conn.commit()

        try:
            return render_template('result.html', prediction_text=prediction_text)
        except Exception as e:
            print(f"Error: {str(e)}")
            return "An error occurred while making predictions."

if __name__ == '__main__':
    app.run(port=os.getenv('PORT', 5000), debug=True)
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load your saved scaler and model 
scaler = joblib.load('transformer.save')          
model = joblib.load('random_forest_model.save')      

@app.route('/', methods=['GET'])
def home():
    return render_template('Manual_predict.html')

@app.route('/y_predict', methods=['POST'])
def y_predict():
    try:
        # Get the 7 input values from the form 
        # Order should match your X columns: u_q, coolant, u_d, motor_speed, i_d, i_q, ambient
        features = [
            float(request.form['u_q']),
            float(request.form['coolant']),
            float(request.form['u_d']),
            float(request.form['motor_speed']),
            float(request.form['i_d']),
            float(request.form['i_q']),
            float(request.form['ambient'])
        ]

        # Convert to 2D array (1 sample)
        x_test = np.array([features])

        # Scale using the loaded scaler
        x_test_scaled = scaler.transform(x_test)

        # Predict
        prediction = model.predict(x_test_scaled)[0]

        # Format the result (you can adjust the message)
        prediction_text = f'Permanent Magnet surface temperature: {prediction:.4f}'

        # For debugging in terminal
        print("Input features:", features)
        print("Scaled:", x_test_scaled)
        print("Prediction:", prediction)

        return render_template('Manual_predict.html', prediction_text=prediction_text)

    except Exception as e:
        error_text = f"Error: {str(e)}. Please check all inputs are numbers."
        return render_template('Manual_predict.html', prediction_text=error_text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)  
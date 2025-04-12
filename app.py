from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Explicitly set static folder
app = Flask(__name__, static_folder='static')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        float_features = [float(x) for x in request.form.values()]
        features = [np.array(float_features)]
        prediction = model.predict(features)
        return render_template("index.html", prediction_text=f"The Predicted Price is ${prediction[0]:,.2f}")
    except ValueError:
        return render_template("index.html", prediction_text="Invalid input! Please enter valid numbers.")

if __name__ == "__main__":
    app.run(debug=True)

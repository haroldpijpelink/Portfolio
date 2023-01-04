from flask import Flask, request, jsonify
from sklearn.externals import joblib

# Load the model
model = joblib.load("model.pkl")

# Create an instance of the Flask class
app = Flask(__name__)

# Define a route for the default URL
@app.route('/')
def home():
    return "Welcome to the data science application!"

# Define a route for the prediction API
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)
    
    # Make the prediction
    prediction = model.predict(data)
    
    # Return the prediction as a response
    return jsonify(prediction)

# Run the app
if __name__ == '__main__':
    app.run(port=5000, debug=True)

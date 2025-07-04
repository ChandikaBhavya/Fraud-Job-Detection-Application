from flask import Flask, request, jsonify
import joblib

# Initialize the Flask application
app = Flask(__name__)

# Load the trained Logistic Regression model
model = joblib.load('LogisticRegression_model.pkl')

# Load the CountVectorizer (which you saved earlier)
cv = joblib.load('count_vectorizer.pkl')

# Home route for testing the server
@app.route('/')
def home():
    return "Welcome to the Logistic Regression Model API!"

# Predict route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input JSON data
    data = request.get_json()
    
    # Extract the 'text' field from the input data
    text = data['text']
    
    # Transform the text using the saved CountVectorizer
    text_vectorized = cv.transform([text])
    
    # Make a prediction using the trained model
    prediction = model.predict(text_vectorized)
    
    # Return the prediction result as a JSON response
    return jsonify({'prediction': int(prediction[0])})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

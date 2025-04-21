from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

app = Flask(__name__)
CORS(app)

# Load the trained models and tokenizer
full_lstm_model = load_model('lstm_model.keras')  # Updated to load .keras format
xgb_model = joblib.load('xgboost_model.pkl')
tokenizer = joblib.load('tokenizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

MAX_LEN = 50

# Extract features from LSTM up to the Dense(128) layer
lstm_feature_extractor = Sequential(full_lstm_model.layers[:-1])  # Remove the final softmax layer

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("query", "")

    if not text:
        return jsonify({"error": "Query text is required"}), 400

    # Tokenize and pad the input
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding="post")

    # Extract LSTM features
    lstm_features = lstm_feature_extractor.predict(padded)

    # Predict category using XGBoost
    xgb_prediction = xgb_model.predict(lstm_features)
    predicted_category = label_encoder.inverse_transform(xgb_prediction)[0]

    return jsonify({"query": text, "predicted_category": predicted_category})


if __name__ == "__main__":
    app.run(debug=True) 
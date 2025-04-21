import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from xgboost import XGBClassifier
import joblib

# Step 1: Load Dataset
def load_dataset(file_path):
    # Load the Excel file
    df = pd.read_excel(file_path)

    # Display basic information about the dataset
    print("Dataset Overview:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())

    return df

# Step 2: Preprocess Data
def preprocess_data(df):
    # Remove missing values
    df.dropna(inplace=True)

    # Convert queries to lowercase
    df['utterance'] = df['utterance'].str.lower()

    # Encode labels
    label_encoder = LabelEncoder()
    df['category_encoded'] = label_encoder.fit_transform(df['category'])

    # Save the label encoder for later use
    joblib.dump(label_encoder, 'label_encoder.pkl')

    # Tokenize and pad sequences
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['utterance'])
    sequences = tokenizer.texts_to_sequences(df['utterance'])
    padded_sequences = pad_sequences(sequences, maxlen=50, padding='post', truncating='post')

    # Save the tokenizer for later use
    joblib.dump(tokenizer, 'tokenizer.pkl')

    return padded_sequences, df['category_encoded'], tokenizer, label_encoder

# Step 3: Train LSTM + XGBoost Model
def train_model(X_train, X_val, y_train, y_val, tokenizer, num_classes):
    # Load GloVe embeddings
    embedding_index = {}
    with open("glove.6B.200d.txt", encoding="utf-8") as f:  # Update with the correct GloVe path
        for line in f:
            values = line.split()
            word = values[0]
            coefficients = np.asarray(values[1:], dtype="float32")
            embedding_index[word] = coefficients

    # Create embedding matrix
    embedding_matrix = np.zeros((5000, 200))
    for word, i in tokenizer.word_index.items():
        if i < 5000:
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    # Define the LSTM model
    model = Sequential([
        Embedding(input_dim=5000, output_dim=200, input_length=50, weights=[embedding_matrix], trainable=False),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(128, activation='relu'),  # Ensure this matches XGBoost expected input shape
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    optimizer = Adam(learning_rate=0.0001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Compute class weights
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(class_weights))

    # Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        class_weight=class_weights,
        callbacks=[early_stopping]
    )

    # Save the LSTM model
    model.save('lstm_model.keras')

    # Extract features using the LSTM model
    feature_extractor = Sequential(model.layers[:-1])  # Remove the softmax layer
    X_train_features = feature_extractor.predict(X_train)
    X_val_features = feature_extractor.predict(X_val)

    # Train the XGBoost classifier
    xgb_model = XGBClassifier(
        eval_metric='mlogloss',
        learning_rate=0.1,
        max_depth=6,
        n_estimators=100
    )
    xgb_model.fit(X_train_features, y_train)

    # Save the XGBoost model
    joblib.dump(xgb_model, 'xgboost_model.pkl')

    return model, xgb_model
# Main Script
if __name__ == "__main__":
    # Step 1: Load the dataset
    file_path = "NLP Project A dataset.xlsx"  # Update with the correct file path
    df = load_dataset(file_path)

    # Step 2: Preprocess the data
    padded_sequences, encoded_labels, tokenizer, label_encoder = preprocess_data(df)

    # Step 3: Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        padded_sequences, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )

    # Step 4: Train the model
    lstm_model, xgb_model = train_model(X_train, X_val, y_train, y_val, tokenizer, len(label_encoder.classes_))
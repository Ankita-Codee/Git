# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import matplotlib.pyplot as plt
import streamlit as st
import os
from datetime import datetime

# Constants
MAX_WORDS = 10000
MAX_LEN = 100
EMBEDDING_DIM = 100
BATCH_SIZE = 32
EPOCHS = 10

def load_and_preprocess_data():
    """Load and preprocess the IMDB dataset."""
    # Load the dataset
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_WORDS)
    
    # Pad sequences
    x_train = pad_sequences(x_train, maxlen=MAX_LEN)
    x_test = pad_sequences(x_test, maxlen=MAX_LEN)
    
    return (x_train, y_train), (x_test, y_test)

def create_model():
    """Create and compile the Simple RNN model."""
    model = Sequential([
        Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),
        SimpleRNN(32, return_sequences=True),
        Dropout(0.2),
        SimpleRNN(32),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, x_train, y_train):
    """Train the model with callbacks."""
    # Create callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    # Create TensorBoard callback
    log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        callbacks=[early_stopping, tensorboard_callback]
    )
    
    return history

def plot_training_history(history):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def predict_sentiment(model, text):
    """Predict sentiment for a given text."""
    # Convert text to sequence
    word_index = imdb.get_word_index()
    sequence = []
    for word in text.lower().split():
        if word in word_index and word_index[word] < MAX_WORDS:
            sequence.append(word_index[word])
    
    # Pad sequence
    padded_sequence = pad_sequences([sequence], maxlen=MAX_LEN)
    
    # Make prediction
    prediction = model.predict(padded_sequence)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    
    return sentiment, prediction

def main():
    st.title("IMDB Sentiment Analysis with Simple RNN")
    
    # Load and preprocess data
    st.write("Loading and preprocessing data...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Create and train model
    st.write("Creating and training model...")
    model = create_model()
    history = train_model(model, x_train, y_train)
    
    # Plot training history
    st.write("Training History:")
    fig = plot_training_history(history)
    st.pyplot(fig)
    
    # Evaluate model
    st.write("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    st.write(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Save model
    model.save('simple_rnn_imdb.h5')
    st.write("Model saved as 'simple_rnn_imdb.h5'")
    
    # Create prediction interface
    st.write("## Try it yourself!")
    text_input = st.text_area("Enter a movie review:", "")
    if st.button("Predict Sentiment"):
        if text_input:
            sentiment, confidence = predict_sentiment(model, text_input)
            st.write(f"Predicted Sentiment: {sentiment}")
            st.write(f"Confidence: {confidence:.4f}")
        else:
            st.write("Please enter some text!")

if __name__ == "__main__":
    main()


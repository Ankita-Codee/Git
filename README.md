# IMDB Sentiment Analysis with Simple RNN

This project implements a sentiment analysis model using a Simple Recurrent Neural Network (RNN) on the IMDB movie reviews dataset. The model classifies movie reviews as positive or negative.

## Project Structure

- `main.py`: Main application file containing the model implementation and Streamlit interface
- `requirements.txt`: List of Python dependencies
- `simple_rnn_imdb.h5`: Trained model file (generated after training)

## Features

- Data preprocessing and tokenization
- Simple RNN model with embedding layer
- Training with early stopping and TensorBoard integration
- Interactive web interface using Streamlit
- Real-time sentiment prediction

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run main.py
```

2. The application will:
   - Load and preprocess the IMDB dataset
   - Train the Simple RNN model
   - Display training history plots
   - Show model evaluation metrics
   - Provide an interface for making predictions on new reviews

## Model Architecture

- Embedding Layer (10000 words, 100 dimensions)
- Simple RNN Layer (32 units, return sequences)
- Dropout Layer (0.2)
- Simple RNN Layer (32 units)
- Dropout Layer (0.2)
- Dense Layer (1 unit, sigmoid activation)

## Training Parameters

- Maximum words: 10000
- Maximum sequence length: 100
- Batch size: 32
- Epochs: 10
- Early stopping patience: 3

## Performance

The model typically achieves:
- Training accuracy: ~85-90%
- Validation accuracy: ~80-85%
- Test accuracy: ~80-85%

## Contributing

Feel free to submit issues and enhancement requests! 
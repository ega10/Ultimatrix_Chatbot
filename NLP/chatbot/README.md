# ULTIMATRIX CHATBOT

This project implements a **Chatbot** using Natural Language Processing (NLP) and Machine Learning. It uses a pre-trained neural network model to classify user queries and provide appropriate responses. The model is trained using the Keras library with TensorFlow as the backend. The project also makes use of other libraries such as NumPy, NLTK, and Streamlit for building the user interface.

## Features
- **Intent Classification**: The chatbot classifies user queries into predefined intents and provides relevant responses.
- **Natural Language Understanding**: The chatbot uses NLTK for tokenizing and lemmatizing text to better understand user input.
- **Pre-trained Model**: A neural network model is used for predicting the intent behind user input.
- **Interactive Interface**: The chatbot can be interacted with via a simple user interface.

## Installation

### Prerequisites
- Python 3.7 or higher
- TensorFlow 2.x (Including Keras)
- Numpy
- NLTK
- pybind11 (If using NumPy 2.x)

### Step 1: Clone the Repository
### Step 2: Install Dependencies
### Step 3: Download Required Files
Ensure the following files are in the same directory as the script:

chatbot_model1.h5: The pre-trained Keras model for the chatbot.
words.pkl: List of words used in the model.
classes.pkl: List of intents that the model predicts.
intents.json: JSON file containing intent-response pairs for the chatbot.

### Libraries Used
TensorFlow: Machine learning framework for deep learning models.
Keras: High-level neural networks API for building and training models.
NumPy: Library for numerical operations and array manipulation.
NLTK: Natural Language Toolkit for text preprocessing, tokenization, and lemmatization.
Streamlit: For building interactive web apps (if used).

import random
import numpy as np
import pickle
import json
import streamlit as st
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

# Initialize necessary components
lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')
model = load_model(r"C:\Users\egadarshan\Desktop\NLP\chatbot\chatbot_model.h5")
words = pickle.load(open(r"C:\Users\egadarshan\Desktop\NLP\chatbot\words.pkl", "rb"))
classes = pickle.load(open(r"C:\Users\egadarshan\Desktop\NLP\chatbot\classes.pkl", "rb"))

# Load the intents file
with open(r"C:\Users\egadarshan\Desktop\NLP\chatbot\intents.json") as file:
    intents = json.load(file)

# Clean up sentence for tokenization
def clean_up_sentence(sentence):
    sentence_words = tokenizer.tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Convert sentence into a bag of words
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Predict the intent of the sentence
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list

# Get response from intents file
def getResponse(ints, intents_json):
    if not ints:
        return "No valid intent found."
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            return random.choice(i["responses"])
    return "Sorry, I don't understand that."

# Define the chatbot response
def chatbot_response(user_input):
    if user_input.startswith('my name is'):
        name = user_input[11:]
        ints = predict_class(user_input, model)
        res = getResponse(ints, intents)
        return res.replace("{n}", name)
    elif user_input.startswith('hi my name is'):
        name = user_input[14:]
        ints = predict_class(user_input, model)
        res = getResponse(ints, intents)
        return res.replace("{n}", name)
    else:
        ints = predict_class(user_input, model)
        return getResponse(ints, intents)

# Streamlit interface
st.title("Chatbot Interface")
st.markdown("Welcome to the chatbot! Ask me anything.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Input box
user_input = st.text_input("You:", "")

if user_input:
    # Display the user's input
    st.session_state.messages.append({"role": "user", "message": user_input})

    # Get the bot's response
    bot_response = chatbot_response(user_input)
    st.session_state.messages.append({"role": "bot", "message": bot_response})

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['message']}")
    else:
        st.markdown(f"**Bot:** {message['message']}")

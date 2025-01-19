import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from textblob import TextBlob

# Load pre-trained model and tokenizer
MODEL_NAME = "Breakfast01/chinese-english-xlm-r"  # Replace with your model's name or your Hugging Face model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Streamlit interface
st.title("Hugging Face Model with Streamlit")

# Input text for classification
input_text = st.text_area("Enter Text for Classification:")

# Button to make prediction
if st.button("Classify"):
    if input_text:
        # Tokenize the input
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Get model outputs
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the predicted class
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1)

        # Map to actual labels (if known)
        class_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}  # Adjust based on your model
        predicted_label = class_labels[predicted_class.item()]

        # Polarity and Subjectivity using TextBlob
        blob = TextBlob(input_text)
        polarity = blob.sentiment.polarity  # Range [-1, 1], -1 = negative, 1 = positive
        subjectivity = blob.sentiment.subjectivity  # Range [0, 1], 0 = objective, 1 = subjective
        # Display the result
        st.write(f"Predicted Label: {predicted_label}")
    else:
        st.write("Please enter some text for classification.")

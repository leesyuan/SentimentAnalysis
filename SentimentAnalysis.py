import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from textblob import TextBlob
import langid
from snownlp import SnowNLP

# Load pre-trained model and tokenizer
MODEL_NAME = "Breakfast01/chinese-english-xlm-r"  # Replace with your model's name or your Hugging Face model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Function to detect and split text into English and Chinese
def split_text(text):
    english_text = []
    chinese_text = []
    
    # Split the input text into words and classify language for each word
    for word in text.split():
        lang, _ = langid.classify(word)
        if lang == "en":
            english_text.append(word)
        elif lang == "zh":
            chinese_text.append(word)

    return " ".join(english_text), " ".join(chinese_text)
    
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

        # Display the result
        st.write(f"Predicted Label: {predicted_label}")
        english_part, chinese_part = split_text(input_text)

        # Analyze English part with TextBlob
        if english_part:
            blob = TextBlob(english_part)
            english_polarity = blob.sentiment.polarity
            english_subjectivity = blob.sentiment.subjectivity
            st.write(f"English Polarity: {english_polarity:.2f}")
            st.write(f"English Subjectivity: {english_subjectivity:.2f}")
        else:
            st.write("No English text detected.")

        # Analyze Chinese part with SnowNLP
        if chinese_part:
            s = SnowNLP(chinese_part)
            chinese_polarity = s.sentiments  # Polarity: range [0, 1] where 0 is negative, 1 is positive
            chinese_subjectivity = 0  # SnowNLP does not provide subjectivity
            st.write(f"Chinese Polarity: {chinese_polarity:.2f}")
            st.write(f"Chinese Subjectivity: {chinese_subjectivity:.2f}")
        else:
            st.write("No Chinese text detected.")
    
    else:
        st.write("Please enter some text for classification.")

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Load model and tokenizer
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, output_attentions=True)

# Streamlit interface
st.title("Hugging Face Sentiment Classification with Attention Visualization")

# Input text
input_text = st.text_area("Enter text for sentiment classification:")

# Button to make prediction
if st.button("Classify"):
    if input_text:
        # Tokenize the input
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Get model outputs and attention weights
        with torch.no_grad():
            outputs = model(**inputs)

        # Predicted class
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1)
        class_labels = {0: "Negative", 1: "Positive"}
        predicted_label = class_labels[predicted_class.item()]

        # Extract attention weights
        attentions = outputs.attentions[-1]
        attention_weights = attentions[0][0].detach().numpy()
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Identify important tokens based on attention
        important_tokens = []
        threshold = 0.5
        for idx, token in enumerate(tokens):
            if np.max(attention_weights[idx]) > threshold:
                important_tokens.append(token)

        # Display results
        st.write(f"Predicted Label: {predicted_label}")
        st.write(f"Important Tokens: {', '.join(important_tokens)}")
    else:
        st.write("Please enter text for classification.")

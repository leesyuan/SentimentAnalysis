import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import matplotlib.pyplot as plt

# Load pre-trained model and tokenizer
MODEL_NAME = "Breakfast01/chinese-english-xlm-r"  # Adjust with your model name
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, output_attentions=True)

# Streamlit interface
st.title("Multilingual Sentiment Classification with Attention Visualization")

# Input text for sentiment classification
input_text = st.text_area("Enter text for sentiment classification:")

# Button to make prediction
if st.button("Classify"):
    if input_text:
        # Tokenize the input
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Get model outputs and attention weights
        with torch.no_grad():
            outputs = model(**inputs)

        # Get predicted class
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1)
        class_labels = {0: "Negative", 1: "Positive"}  # Adjust labels based on your model
        predicted_label = class_labels[predicted_class.item()]

        # Extract attention weights from the last layer
        attentions = outputs.attentions[-1]  # Get attention from the last layer
        attention_weights = attentions[0][0].detach().numpy()  # First example, first attention head
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Identify important tokens based on attention weights
        important_tokens = []
        threshold = 0.5  # Adjust this threshold based on your model’s attention weights
        for idx, token in enumerate(tokens):
            if np.max(attention_weights[idx]) > threshold:
                important_tokens.append(token)

        # Display prediction results
        st.write(f"Predicted Label: {predicted_label}")
        st.write(f"Important Tokens: {', '.join(important_tokens)}")

        # Plot the attention heatmap
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.matshow(attention_weights, cmap='viridis')

        # Add labels to the matrix
        ax.set_xticks(np.arange(len(tokens)))
        ax.set_yticks(np.arange(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90)
        ax.set_yticklabels(tokens)

        plt.title("Attention Heatmap")
        st.pyplot(fig)
    else:
        st.write("Please enter text for classification.")

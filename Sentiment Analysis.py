{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "6d9ca556-5968-4a0f-b5ca-467f06ec393d",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/conda/lib/python3.11/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html\n",
      "  from .autonotebook import tqdm as notebook_tqdm\n",
      "2025-01-19 08:03:01.205717: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered\n",
      "WARNING: All log messages before absl::InitializeLog() is called are written to STDERR\n",
      "E0000 00:00:1737273781.220186 3510592 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered\n",
      "E0000 00:00:1737273781.224586 3510592 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered\n",
      "2025-01-19 08:03:01.239894: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.\n",
      "To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predicted Class Index: 0\n",
      "Predicted Label: Negative\n"
     ]
    }
   ],
   "source": [
    "from transformers import AutoTokenizer, AutoModelForSequenceClassification\n",
    "import torch\n",
    "\n",
    "# Load pre-trained model and tokenizer\n",
    "MODEL_NAME = \"chinese_english_xlm_r\"  # You can choose a different model\n",
    "tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)\n",
    "model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)\n",
    "\n",
    "# Sample text to classify\n",
    "text = \"不值得这个价格，It's not worth the price.\"\n",
    "\n",
    "# Tokenize the text\n",
    "inputs = tokenizer(text, return_tensors=\"pt\", padding=True, truncation=True, max_length=512)\n",
    "\n",
    "# Get model outputs\n",
    "with torch.no_grad():\n",
    "    outputs = model(**inputs)\n",
    "\n",
    "# Get the predicted class\n",
    "logits = outputs.logits\n",
    "predicted_class = torch.argmax(logits, dim=-1)\n",
    "\n",
    "# Map to the actual class labels (if known)\n",
    "class_labels = {0: \"Negative\", 1: \"Positive\"}  # Customize based on your model's labels\n",
    "predicted_label = class_labels[predicted_class.item()]\n",
    "\n",
    "# Print results\n",
    "print(f\"Predicted Class Index: {predicted_class.item()}\")\n",
    "print(f\"Predicted Label: {predicted_label}\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

# Import libraries
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import numpy as np #hi

# 1. Load the tokenizer and pre-trained RoBERTa model for emotion analysis
tokenizer = AutoTokenizer.from_pretrained("pysentimiento/robertuito-emotion-analysis")
model = AutoModelForSequenceClassification.from_pretrained("pysentimiento/robertuito-emotion-analysis")

# 2. Sample texts to analyze
texts = [
    "I just got an amazing gift! I'm so happy!",
    "I'm feeling really sad today...",
    "This is so frustrating! I can't believe it.",
    "I love hanging out with my friends!"
]

# 3. Get emotion labels from the model configuration (print(labels) to see the different labels)
labels = model.config.id2label

# 4. Predict emotions for each text
for t in texts:
    # Tokenize text into numbers the model understands
    inputs = tokenizer(t, return_tensors="pt")
    
    # Run the model to get raw scores (logits)
    outputs = model(**inputs)
    
    # Convert logits to probabilities
    probs = torch.softmax(outputs.logits, dim=1)
    
    # Get top 3 emotions and their probabilities
    top3 = torch.topk(probs, k=3)
    top3_probs = top3.values[0]
    top3_indices = top3.indices[0]
    top3_emotions = [labels[i.item()] for i in top3_indices]
    
    # Print top 3 emotions for the text
    print(f"Text: {t}")
    for i in range(3):
        print(f"  {i+1}. {top3_emotions[i]} ({top3_probs[i].item()*100:.2f}%)")
    print("\n")

# Load CSV dataset, so the texts and labels can be used
df = pd.read_csv("goemotions_1.csv")

# keep only text and emotion columns
emotion_cols = df.columns[9:] # all the emotion columns
# turns the emotion columns into a list -> combines both text and emotion lists
df = df[["text"] + list(emotion_cols)] 
# print(df.head())

# ---------------------------
# BATCH TESTING (recommended)
# ---------------------------
batch_size = 32
predictions = []

for i in range(0, len(df), batch_size):
    batch_texts = df["text"].iloc[i:i+batch_size].tolist()  # get a batch
    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    predictions.extend(probs.detach().numpy())  # add batch predictions

# convert list to array for easier processing
predictions = np.array(predictions)

# get the actual labels from your dataset
true_labels = df[emotion_cols].astype(int).values

# ---------------------------
# EVERYTHING AT ONCE (commented out)
# ---------------------------
"""
# predictions will hold the probabilities for each emotion for every text
predictions = []

# predict emotions for each text (single-text loop)
for t in df["text"]:
    inputs = tokenizer(t, return_tensors="pt")
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    predictions.append(probs.detach().numpy()[0])  # .detach().numpy() converts tensor to normal array for easier processing
predictions = np.array(predictions) # convert list to array
"""

# ---------------------------
# Compare predicted vs actual labels (example: top-1 accuracy)
# ---------------------------
top_k = 3
correct_count = 0

for i in range(len(true_labels)):
    topk_preds = np.argsort(predictions[i])[::-1][:top_k]  # top 3 predicted emotions
    if any(true_labels[i][idx] == 1 for idx in topk_preds):  # match any true emotion
        correct_count += 1

top3_accuracy = correct_count / len(true_labels)
print(f"Top-{top_k} accuracy on dataset: {top3_accuracy*100:.2f}%")


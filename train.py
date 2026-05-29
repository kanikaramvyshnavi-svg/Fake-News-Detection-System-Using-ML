import pandas as pd
import pickle
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load datasets
fake = pd.read_csv("Fake.csv", encoding="latin1")
true = pd.read_csv("True.csv", encoding="latin1")

# Labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
data = pd.concat([fake, true])

# Shuffle dataset
data = data.sample(frac=1, random_state=42)

# Combine title and text
data["content"] = data["title"] + " " + data["text"]

# CLEANING FUNCTION
def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)

    return text

# Apply cleaning
data["content"] = data["content"].apply(clean_text)

# Features and labels
x = data["content"]
y = data["label"]

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

xv_train = vectorizer.fit_transform(x_train)
xv_test = vectorizer.transform(x_test)

# Train model
model = LogisticRegression(max_iter=1000)

model.fit(xv_train, y_train)

# Prediction
pred = model.predict(xv_test)

# Accuracy
score = accuracy_score(y_test, pred)

print("Accuracy:", score * 100)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

# Save vectorizer
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model Saved Successfully")
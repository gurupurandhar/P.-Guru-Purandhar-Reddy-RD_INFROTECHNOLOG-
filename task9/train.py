import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset (Sentiment140)
data = pd.read_csv(
    "sentiment.csv",
    encoding="latin-1",
    header=None
)

# Give column names
data.columns = [
    "sentiment",
    "id",
    "date",
    "query",
    "user",
    "text"
]

# Keep only required columns
data = data[["text", "sentiment"]]

# Convert labels
data["sentiment"] = data["sentiment"].replace({
    0: "Negative",
    2: "Neutral",
    4: "Positive"
})

# Convert text into vectors
vectorizer = TfidfVectorizer(stop_words="english")

X = vectorizer.fit_transform(data["text"])

y = data["sentiment"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()

model.fit(X_train, y_train)

# Accuracy
prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Save model
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Training Completed Successfully!")
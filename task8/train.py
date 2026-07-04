import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# Merge datasets
movies = movies.merge(credits, on="title")

# Keep required columns
movies = movies[["movie_id", "title", "overview"]]

# Remove empty rows
movies.dropna(inplace=True)

# Convert overview text into vectors
cv = CountVectorizer(max_features=5000, stop_words="english")

vectors = cv.fit_transform(movies["overview"]).toarray()

# Calculate similarity
similarity = cosine_similarity(vectors)

# Save files
joblib.dump(movies, "movie_list.pkl")
joblib.dump(similarity, "similarity.pkl")

print("Training Completed Successfully!")
print("movie_list.pkl created")
print("similarity.pkl created")
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from rapidfuzz import process

df = pd.read_csv("movie_dataset.csv")

# Data cleaning
df['rating'] = df['rating'].fillna(df['rating'].mean())
df['genre'] = df['genre'].fillna('')
df['release_date'] = df['release_date'].fillna('Unknown')
df['overview'] = df['overview'].fillna('Unknown')

df['genre'] = (
    df['genre']
    .str.replace('Science Fiction', 'Sci-Fi', regex=False)
    .str.replace(',', ' ', regex=False)
)

# TF-IDF model (ONE MODEL)
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['genre'])

indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# Format output
def _format_result(result_df):
    output = result_df.copy()
    # output['imdb_id'] = output['imdb_id'].fillna('Unknown')
    return output.to_dict(orient='records')

# Recommendation: By Movie
def recommend_by_movie(movie_title, top_n=10):
    if movie_title not in indices:
        suggestions = process.extract(movie_title, indices.index, limit=5)
        return {
            "error": "Movie not found",
            "did_you_mean": [s[0] for s in suggestions]
        }

    idx = indices[movie_title]
    if isinstance(idx, (pd.Series, list, np.ndarray)):
        idx = int(idx.iloc[0]) if hasattr(idx, "iloc") else int(idx[0])
    else:
        idx = int(idx)

    similarity_scores = linear_kernel(
        tfidf_matrix[idx:idx+1],
        tfidf_matrix
    ).flatten()

    similarity_scores[idx] = 0

    temp_df = df.assign(similarity=similarity_scores)
    temp_df = temp_df[temp_df['similarity'] > 0]
    
    # Get top candidates (3x the required amount) sorted by rating
    candidates = temp_df.sort_values(by='rating', ascending=False).head(top_n * 3)
    
    # Randomly sample from top candidates
    if len(candidates) >= top_n:
        result = candidates.sample(n=top_n, random_state=None)
    else:
        result = candidates
    
    result = result[['title', 'genre', 'rating', 'release_date', 'overview']]

    return _format_result(result)

# Recommendation: By Genre
def recommend_by_genre(genre, top_n=10):
    if not genre:
        return []

    # Handle both single genre (string) and multiple genre (list)
    if isinstance(genre, str):
        genre = [genre]
    
    # Replace Science Fiction with Sci-Fi for all genre
    genre = [g.replace("Science Fiction", "Sci-Fi") for g in genre]

    # Create mask that matches ALL selected genre (AND logic)
    genre_mask = pd.Series([True] * len(df), index=df.index)
    for genre in genre:
        genre_mask &= df['genre'].str.contains(genre, case=False, na=False, regex=False)
    
    # Get top candidates (3x the required amount) sorted by rating
    candidates = (
        df[genre_mask]
        .sort_values(by='rating', ascending=False)
        .head(top_n * 3)
    )
    
    # Randomly sample from top candidates
    if len(candidates) >= top_n:
        result = candidates.sample(n=top_n, random_state=None)
    else:
        result = candidates
    
    result = result[['title', 'genre', 'rating', 'release_date', 'overview']]

    return _format_result(result)

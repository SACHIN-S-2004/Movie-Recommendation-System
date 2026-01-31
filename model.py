import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from rapidfuzz import process

df = pd.read_csv("movie_dataset_cleaned.csv")

# Data cleaning
df['rating'] = df['rating'].fillna(df['rating'].mean())
df['genres'] = df['genres'].fillna('')
df['release_date'] = pd.to_numeric(df['release_date'], errors='coerce')

df['genres'] = (
    df['genres']
    .str.replace('Science Fiction', 'Sci-Fi', regex=False)
    .str.replace(',', ' ', regex=False)
)

# TF-IDF model (ONE MODEL)
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['genres'])

indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# Format output
def _format_result(result_df):
    output = result_df.copy()
    output['release_date'] = output['release_date'].fillna('Unknown')
    output['imdb_id'] = output['imdb_id'].fillna('Unknown')
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
    temp_df = temp_df.sort_values(by='rating', ascending=False)

    result = temp_df[['title', 'genres', 'rating', 'release_date', 'imdb_id']].head(top_n)

    return _format_result(result)

# Recommendation: By Genre
def recommend_by_genre(genre, top_n=10):
    if not genre:
        return []

    genre = genre.replace("Science Fiction", "Sci-Fi")

    genre_mask = df['genres'].str.contains(genre, case=False, na=False)
    result = (
        df[genre_mask]
        .sort_values(by='rating', ascending=False)
        [['title', 'genres', 'rating', 'release_date','imdb_id']]
        .head(top_n)
    )

    return _format_result(result)

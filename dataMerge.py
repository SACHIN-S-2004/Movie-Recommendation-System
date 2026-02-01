# import pandas as pd

# df = pd.read_csv("top10K-TMDB-movies.csv")

# df = df.rename(columns={'vote_average': 'rating'})
# selected_columns = ["id","title","genre","overview","rating","release_date"]
# df_filtered = df[selected_columns]

# df_filtered.to_csv("movie_dataset.csv", index=False)

import pandas as pd

df = pd.read_csv("movie_dataset.csv")

df["genre"] = df["genre"].str.replace("Science Fiction", "Sci-Fi", regex=False)

df.to_csv("movie_dataset_cleaned.csv", index=False)

# import pandas as pd

# df = pd.read_csv("updated_file.csv")

# selected_columns = ["id","title","rating","release_date","adult","imdb_id","overview","genres"]   # replace with your column names
# df_filtered = df[selected_columns]

# df_filtered.to_csv("movie_dataset.csv", index=False)

import pandas as pd

df = pd.read_csv("movie_dataset.csv")

df["genres"] = df["genres"].str.replace("Science Fiction", "Sci-Fi", regex=False)

df.to_csv("movie_dataset_cleaned.csv", index=False)

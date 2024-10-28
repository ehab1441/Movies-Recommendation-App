import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import random
import re

# Load the movies data (correcting the path)
movies = pd.read_csv(r"movies_data.csv", lineterminator='\n')

def extract_years(df):
    df['Release_Year'] = pd.to_datetime(df['Release_Year'], errors='coerce').dt.year
    return df

def map_language_codes_to_full_names(df):
    language_map = {
        'en': 'English', 'ja': 'Japanese', 'fr': 'French', 'hi': 'Hindi', 'es': 'Spanish',
        'ru': 'Russian', 'de': 'German', 'th': 'Thai', 'ko': 'Korean', 'tr': 'Turkish',
        'cn': 'Chinese', 'zh': 'Chinese', 'it': 'Italian', 'pt': 'Portuguese', 'ml': 'Malayalam',
        'pl': 'Polish', 'fi': 'Finnish', 'no': 'Norwegian', 'da': 'Danish', 'id': 'Indonesian',
        'sv': 'Swedish', 'nl': 'Dutch', 'te': 'Telugu', 'sr': 'Serbian', 'is': 'Icelandic',
        'ro': 'Romanian', 'tl': 'Tagalog', 'fa': 'Persian', 'uk': 'Ukrainian', 'nb': 'Norwegian Bokmål',
        'eu': 'Basque', 'lv': 'Latvian', 'ar': 'Arabic', 'el': 'Greek', 'cs': 'Czech', 'ms': 'Malay',
        'bn': 'Bengali', 'ca': 'Catalan', 'la': 'Latin', 'ta': 'Tamil', 'hu': 'Hungarian', 
        'he': 'Hebrew', 'et': 'Estonian'
    }
    df['Original_Language_Full'] = df['Original_Language'].map(language_map)
    df.drop(columns=['Original_Language'], inplace=True)
    return df

def categorize_year(df):
    bins = [1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, df['Release_Year'].max()]
    labels = [1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]
    df['Release_Era'] = pd.cut(df['Release_Year'], bins=bins, labels=labels, include_lowest=True).astype(int)
    return df

def encode_language(df, label_encoder):
    df['Language_Encoded'] = label_encoder.fit_transform(df['Original_Language_Full'])
    return df

def preprocess_genres(df):
    df['Genre_First_Word'] = df['Genre'].str.split().str[0].str.replace(r'[^\w\s]', '', regex=True)
    genre_dummies = df['Genre'].str.get_dummies(sep=',').astype(int)

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    df['Genre_Cluster'] = dbscan.fit_predict(genre_dummies)

    cluster_genre_map = {}
    for cluster in df['Genre_Cluster'].unique():
        genres_in_cluster = genre_dummies[df['Genre_Cluster'] == cluster].sum()
        primary_genre = genres_in_cluster.idxmax() if not genres_in_cluster.empty else None
        cluster_genre_map[cluster] = primary_genre
    df['Primary_Genre'] = df['Genre_Cluster'].map(cluster_genre_map)

    genre_mapping = {
        'Adventure': 'Adventure', 'Mystery': 'Mystery', 'Thriller': 'Thriller', 'Comedy': 'Comedy', 
        'Crime': 'Crime', 'Science Fiction': 'Science Fiction', 'Action': 'Action', 'Drama': 'Drama', 
        'Horror': 'Horror', 'Fantasy': 'Fantasy', 'War': 'War', 'Romance': 'Romance', 
        'Animation': 'Animation', 'History': 'History', 'Music': 'Music', 'Family': 'Family', 
        'Western': 'Western', 'TV Movie': 'TV Movie', 'Documentary': 'Documentary'
    }

    df['Primary_Genre'] = df['Primary_Genre'].map(genre_mapping)
    df['Primary_Genre'].fillna(df['Genre_First_Word'], inplace=True)
    return df

def encode_primary_genre(df, label_encoder):
    df['Genre_Encoded'] = label_encoder.fit_transform(df['Primary_Genre'])
    return df

def pipeline(df):
    label_encoder = LabelEncoder()
    df = extract_years(df)
    df = map_language_codes_to_full_names(df)
    df = categorize_year(df)
    df = encode_language(df, label_encoder)
    df = preprocess_genres(df)
    df = encode_primary_genre(df, label_encoder)
    return df

# Run the pipeline
movies = pipeline(movies)

# Preparing data for similarity calculation
X = movies[['Genre_Encoded', 'Language_Encoded', 'Release_Year', 'Vote_Count', 'Vote_Average', 'Popularity']]
similarity_matrix = cosine_similarity(X)
movie_titles = movies['Title'].values

def recommend_movies(movie_title_encoded, similarity_matrix, movie_titles, top_n=5):
    """
    Recommends top_n movies similar to the given movie title based on cosine similarity.
    """
    # Find the index of the encoded movie title
    movie_indices = np.where(movie_titles == movie_title_encoded)[0]
    
    if len(movie_indices) == 0:
        print(f"Encoded title '{movie_title_encoded}' not found in movie titles.")
        return []
    
    movie_idx = movie_indices[0]
    similar_movies = sorted(
        list(enumerate(similarity_matrix[movie_idx])),
        key=lambda x: x[1],
        reverse=True
    )
    recommendations = [movie_titles[i[0]] for i in similar_movies[1:top_n+1]]  # Skip the first item (itself)
    return recommendations

def get_encoded_title_by_features(genre, language, release_years, movies_df):
    """
    Fetches movies by matching Genre, Language, and Release Year.
    """
    filtered_movies = movies_df[
        (movies_df['Primary_Genre'] == genre) &
        (movies_df['Original_Language_Full'] == language) &
        (movies_df['Release_Year'].isin(release_years))
    ]
    return filtered_movies['Title'].values if not filtered_movies.empty else "No movie found with the specified feature values."

# Example usage
genre_input = 'Action'
language_input = 'English'
release_year_input = [2019, 2020]

encoded_titles = get_encoded_title_by_features(genre_input, language_input, release_year_input, movies)

if isinstance(encoded_titles, np.ndarray) and encoded_titles.size > 0:
    selected_title = random.choice(encoded_titles)
    print("Selected Encoded Title:", selected_title)

    # Get recommendations
    recommended_movies = recommend_movies(selected_title, similarity_matrix, movie_titles, top_n=10)
    print("Recommended Movies:", recommended_movies)

    def get_specific_movie_details_by_encoded_title(title_encoded, movies_df, columns):
        """
        Retrieves specific details for a movie by its encoded title.
        """
        movie_details = movies_df[movies_df['Title'] == title_encoded]
        return movie_details[columns].iloc[0] if not movie_details.empty else None
    
    seen_movies_file = 'seen_movies.txt'

    try:
        with open(seen_movies_file, 'r') as file:
            seen_movies = set(file.read().splitlines())
    except FileNotFoundError:
        seen_movies = set()

    # Filter out seen movies from the recommendations
    unseen_movies = [movie for movie in recommended_movies if movie not in seen_movies]

    # Check if there are unseen movies left
    if unseen_movies:
        # Randomly select one unseen movie
        selected_movie = random.choice(unseen_movies)

        # Get the movie details
        columns_to_retrieve = ['Title', 'Overview', 'Release_Year', 'Genre', 'Vote_Average', 'Poster_Url']
        movie_info = get_specific_movie_details_by_encoded_title(selected_movie, movies, columns_to_retrieve)

        # Print the specific movie details
        if movie_info is not None:
            print("\n" + "=" * 40)
            print(f"🎬 {movie_info['Title']}")
            print("=" * 40)
            print(f"Overview: {movie_info['Overview']}")
            print(f"Release Year: {movie_info['Release_Year']}")
            print(f"Genre: {movie_info['Genre']}")
            print(f"Average Rating: {movie_info['Vote_Average']} ⭐")
            print(f"Poster URL: {movie_info['Poster_Url']}")
            print("=" * 40 + "\n")

            # Mark this movie as seen
            seen_movies.add(selected_movie)

            # Save the updated seen movies to the file
            with open(seen_movies_file, 'w') as file:
                file.write("\n".join(seen_movies))
        else:
            print(f"🚫 No movie found with Title Encoded '{selected_movie}'.")
    else:
        print("🎉 You've seen all recommended movies! Generating new recommendations...")
        # Here you can implement logic to fetch new recommendations or refresh the movie list.
else:
    print("No movies found matching the criteria.")

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import random
import re

movies = pd.read_csv('movies_data.csv', lineterminator='\n')

# Initialize label encoder
label_encoder = LabelEncoder()

# Transformer classes
class ExtractYearTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_col='Release_Date'):
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.date_col] = pd.to_datetime(X[self.date_col], errors='coerce')
        X['Release_Year'] = X[self.date_col].dt.year
        X.drop(columns=[self.date_col], inplace=True)
        return X

class LanguageMappingTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
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
        X['Original_Language_Full'] = X['Original_Language'].map(language_map)
        X.drop(columns=['Original_Language'], inplace=True)
        return X

class YearCategorizerTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        bins = [1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, X['Release_Year'].max()]
        labels = [1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]
        X['Release_Era'] = pd.cut(X['Release_Year'], bins=bins, labels=labels, include_lowest=True)
        X['Release_Era'] = X['Release_Era'].astype(int)
        return X

class EncodeColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoder = LabelEncoder()

    def fit(self, X, y=None):
        self.label_encoder.fit(X['Original_Language_Full'])
        return self

    def transform(self, X):
        X = X.copy()
        X['Language_Encoded'] = self.label_encoder.transform(X['Original_Language_Full'])
        X['Genre_First_Word'] = X['Genre'].str.split().str[0].str.replace(r'[^\w\s]', '', regex=True)
        return X

class EncodeGenreTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        genre_dummies = X['Genre'].str.get_dummies(sep=',').astype(int)
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        genre_clusters = dbscan.fit_predict(genre_dummies)
        X['Genre_Cluster'] = genre_clusters
        
        cluster_genre_map = {}
        for cluster in X['Genre_Cluster'].unique():
            genres_in_cluster = genre_dummies[X['Genre_Cluster'] == cluster].sum()
            primary_genre = genres_in_cluster.idxmax()
            cluster_genre_map[cluster] = primary_genre
        
        X['Primary_Genre'] = X['Genre_Cluster'].map(cluster_genre_map)
        X['Primary_Genre'].fillna(X['Genre_First_Word'], inplace=True)

        genre_mapping = {
            ' Adventure': 'Adventure', ' Mystery': 'Mystery', ' Thriller': 'Thriller', ' Comedy': 'Comedy',
            ' Crime': 'Crime', 'Science Fiction': 'Science Fiction', ' Action': 'Action', ' Drama': 'Drama',
            ' Horror': 'Horror', ' Fantasy': 'Fantasy', ' War': 'War', ' Romance': 'Romance', 
            ' Animation': 'Animation', ' History': 'History', ' Music': 'Music', ' Family': 'Family',
            ' Western': 'Western', ' TV Movie': 'TV Movie', ' Documentary': 'Documentary'
        }
        
        X['Primary_Genre'] = X['Primary_Genre'].map(genre_mapping)
        X['Genre_Encoded'] = label_encoder.fit_transform(X['Primary_Genre'])
        return X

# Construct the pipeline
movies_pipeline = Pipeline([
    ('extract_years', ExtractYearTransformer()),
    ('map_languages', LanguageMappingTransformer()),
    ('categorize_year', YearCategorizerTransformer()),
    ('encode_columns', EncodeColumnsTransformer()),
    ('encode_genre', EncodeGenreTransformer())
])

# Apply the pipeline to the movies DataFrame
movies = movies_pipeline.fit_transform(movies)




X = movies[['Genre_Encoded', 'Language_Encoded', 'Release_Year', 'Vote_Count', 'Vote_Average', 'Popularity']]

similarity_matrix = cosine_similarity(X)

def recommend_movies(movie_title_encoded, similarity_matrix, movie_titles, top_n=5):
    # Find the index of the encoded movie title
    movie_indices = np.where(movie_titles == movie_title_encoded)[0]
    
    # Check if the movie exists in the titles
    if len(movie_indices) == 0:
        print(f"Encoded title '{movie_title_encoded}' not found in movie titles.")
        return []  # Return an empty list if not found

    movie_idx = movie_indices[0]
    
    # Get the similarity scores for the specified movie
    similar_movies = list(enumerate(similarity_matrix[movie_idx]))
    
    # Sort the movies based on similarity score, descending
    similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
    
    # Get the top N recommendations, excluding the movie itself (index 0)
    recommendations = [
        movie_titles[i[0]] for i in similar_movies[1:]  # Exclude the movie itself
    ]
    
    return recommendations[:top_n]  # Return the top N recommendations


movie_titles = movies['Title'].values

def get_encoded_title_by_features(Genre, Language, Release_Year, movies_df):
    
   
    # Filter the DataFrame based on provided feature values
    filtered_movies = movies_df[
        (movies_df['Primary_Genre'] == Genre) &
        (movies_df['Original_Language_Full'] == Language) &
        (movies_df['Release_Year'].isin(Release_Year))
    ]
    
    
    if not filtered_movies.empty:
        return filtered_movies['Title'].values
    else:
        return "No movie found with the specified feature values."

# Example usage
# Replace the values with actual encoded values you want to query
genre_input = 'Action'  # Example value for Genre_Encoded
language_input = 'English'  # Example value for Language_Encoded
release_year_input = [2019, 2020]  # Example value for Release_Era

# Get the encoded titles based on the input values
encoded_titles = get_encoded_title_by_features(
    genre_input,
    language_input,
    release_year_input,
    movies
)

encoded_titles = random.choice(encoded_titles)

print("Encoded Titles:", encoded_titles)


def get_encoded_title_by_features(Genre, Language, Release_Year, movies_df):
    
   
    # Filter the DataFrame based on provided feature values
    filtered_movies = movies_df[
        (movies_df['Primary_Genre'] == Genre) &
        (movies_df['Original_Language_Full'] == Language) &
        (movies_df['Release_Year'].isin(Release_Year))
    ]
    
    
    if not filtered_movies.empty:
        return filtered_movies['Title'].values
    else:
        return "No movie found with the specified feature values."

# Example usage
# Replace the values with actual encoded values you want to query
genre_input = 'Action'  # Example value for Genre_Encoded
language_input = 'English'  # Example value for Language_Encoded
release_year_input = [2019, 2020]  # Example value for Release_Era

# Get the encoded titles based on the input values
encoded_titles = get_encoded_title_by_features(
    genre_input,
    language_input,
    release_year_input,
    movies
)

encoded_titles = random.choice(encoded_titles)

print("Encoded Titles:", encoded_titles)

import random 

some_movie_encoded = encoded_titles

recommended_movies = recommend_movies(some_movie_encoded, similarity_matrix, movie_titles, top_n=1000)

recommended_movies = random.sample(recommended_movies, 10)

print("Recommended Movies:",recommended_movies) 


import pandas as pd

# Assuming movies is your DataFrame with all the relevant columns

def get_specific_movie_details_by_encoded_title(title_encoded, movies_df, columns):
    # Filter the DataFrame to find the row with the specified Title_Encoded
    movie_details = movies_df[movies_df['Title'] == title_encoded]
    
    # Check if any movie details are found
    if not movie_details.empty:
        # Return the specified columns
        return movie_details[columns].iloc[0]  # Return the first matching row as a Series
    else:
        return None  # Return None if no movie is found

# Example usage
# Replace these with actual encoded titles from your dataset
encoded_titles_input = recommended_movies  # Example values for Title_Encoded

# Specify the columns you want to retrieve
columns_to_retrieve = ['Title', 'Overview', 'Release_Year', 'Genre', 'Vote_Average', 'Poster_Url']

# Loop through each encoded title and get the movie details
for title_encoded_input in encoded_titles_input:
    movie_info = get_specific_movie_details_by_encoded_title(title_encoded_input, movies, columns_to_retrieve)
    
    # Print the specific movie details
    if movie_info is not None:
            print("\n" + "="*40)
            print(f"🎬 {movie_info['Title']}")
            print("="*40)
            print(f"Overview: {movie_info['Overview']}")
            print(f"Release Year: {movie_info['Release_Year']}")
            print(f"Genre: {movie_info['Genre']}")
            print(f"Average Rating: {movie_info['Vote_Average']} ⭐")
            print(f"Poster URL: {movie_info['Poster_Url']}")
            print("="*40 + "\n")
    else:
        print(f"🚫 No movie found with Title Encoded '{title_encoded_input}'.")


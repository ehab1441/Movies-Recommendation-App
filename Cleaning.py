import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN
import random
import re

# Load the movies data (correcting the path)
movies = pd.read_csv("your_file_path.csv", lineterminator='\n')

# Function to extract year from the Release_Date column
def extract_years(df):
    # Convert 'Release_Date' to datetime and extract the year
    df['Release_Year'] = pd.to_datetime(df['Release_Date'], errors='coerce').dt.year
    return df

# Function to map language codes (e.g., 'en') to full language names (e.g., 'English')
def map_language_codes_to_full_names(df):
    # Define a dictionary for mapping language codes to full names
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
    # Map language codes to full names
    df['Original_Language_Full'] = df['Original_Language'].map(language_map)
    # Drop the original 'Original_Language' column after mapping
    df.drop(columns=['Original_Language'], inplace=True)
    return df

# Function to categorize release years into defined 'eras' or decades
def categorize_year(df):
    # Define bins and labels for different decades
    bins = [1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, df['Release_Year'].max()]
    labels = [1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]
    # Use pd.cut to assign each movie to a decade
    df['Release_Era'] = pd.cut(df['Release_Year'], bins=bins, labels=labels, include_lowest=True).astype(int)
    return df

# Function to encode full language names into numeric labels
def encode_language(df, label_encoder):
    # Encode 'Original_Language_Full' using LabelEncoder
    df['Language_Encoded'] = label_encoder.fit_transform(df['Original_Language_Full'])
    return df

# Function to preprocess and cluster genres, with extraction of primary genre
def preprocess_genres(df):
    # Extract the first genre word as a fallback in case 'Primary_Genre' is NaN
    df['Genre_First_Word'] = df['Genre'].str.split().str[0].str.replace(r'[^\w\s]', '', regex=True)
    
    # Convert genres into dummy variables (one-hot encoding)
    genre_dummies = df['Genre'].str.get_dummies(sep=',').astype(int)

    # Apply DBSCAN clustering to group similar genres
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    df['Genre_Cluster'] = dbscan.fit_predict(genre_dummies)

    # Determine the primary genre for each cluster based on the most frequent genre
    cluster_genre_map = {}
    for cluster in df['Genre_Cluster'].unique():
        genres_in_cluster = genre_dummies[df['Genre_Cluster'] == cluster].sum()
        primary_genre = genres_in_cluster.idxmax() if not genres_in_cluster.empty else None
        cluster_genre_map[cluster] = primary_genre
    # Map each cluster to its primary genre
    df['Primary_Genre'] = df['Genre_Cluster'].map(cluster_genre_map)

    # Genre mapping to ensure consistency
    genre_mapping = {
        'Adventure': 'Adventure', 'Mystery': 'Mystery', 'Thriller': 'Thriller', 'Comedy': 'Comedy', 
        'Crime': 'Crime', 'Science Fiction': 'Science Fiction', 'Action': 'Action', 'Drama': 'Drama', 
        'Horror': 'Horror', 'Fantasy': 'Fantasy', 'War': 'War', 'Romance': 'Romance', 
        'Animation': 'Animation', 'History': 'History', 'Music': 'Music', 'Family': 'Family', 
        'Western': 'Western', 'TV Movie': 'TV Movie', 'Documentary': 'Documentary'
    }

    # Map 'Primary_Genre' to standardized genres; fill with 'Genre_First_Word' if NaN
    df['Primary_Genre'] = df['Primary_Genre'].map(genre_mapping)
    df['Primary_Genre'].fillna(df['Genre_First_Word'], inplace=True)
    return df

# Function to encode primary genre into numeric labels
def encode_primary_genre(df, label_encoder):
    # Encode 'Primary_Genre' using LabelEncoder
    df['Genre_Encoded'] = label_encoder.fit_transform(df['Primary_Genre'])
    return df

# Pipeline function to run all preprocessing steps
def pipeline(df):
    # Drop duplicate titles to avoid duplicate entries in the dataset
    df = df.drop_duplicates(subset='Title', keep='first')
    # Initialize a LabelEncoder instance
    label_encoder = LabelEncoder()
    # Run each preprocessing function in sequence
    df = extract_years(df)
    df = map_language_codes_to_full_names(df)
    df = categorize_year(df)
    df = encode_language(df, label_encoder)
    df = preprocess_genres(df)
    df = encode_primary_genre(df, label_encoder)
    return df

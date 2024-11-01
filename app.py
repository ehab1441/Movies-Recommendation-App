import streamlit as st
import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity
from Recommendation_System import (
    recommend_movies,
    get_title_by_features,
    get_specific_movie_details_by_title,
)
from Cleaning import pipeline

# Load and process movies data
movies = pd.read_csv('movies.csv', lineterminator='\n')
movies = pipeline(movies)

# Prepare similarity matrix for recommendations
X = movies[['Genre_Encoded', 'Language_Encoded', 'Release_Year', 'Vote_Count', 'Vote_Average', 'Popularity']]
similarity_matrix = cosine_similarity(X)
movie_titles = movies['Title'].values

# Configure Streamlit app page
st.set_page_config(
    page_title="Movie Recommendation App",
    page_icon="🎬",
)

# Title and app description
st.title("Movie Recommendation App")
st.write("Get movie recommendations based on your preferences!")

# Input for user preferences
genre = st.selectbox("Select a genre", movies['Primary_Genre'].unique())
language = st.selectbox("Select a language", movies['Original_Language_Full'].unique())
era_input = st.selectbox("Select Era", movies['Re'].unique())

def get_era_years(era_input):
    """
    Given an era (decade start year), returns a list of years for that decade.
    """
    if not isinstance(era_input, int):
        raise ValueError("Invalid era input. Please provide an integer.")
    era_start = era_input
    era_end = era_start + 9
    return list(range(era_start, era_end + 1))

# Get the decade years based on selected era
years = get_era_years(era_input)

# Button to trigger movie recommendations
if st.button("Recommend a movie"):
    # Find encoded titles that match user selections
    encoded_titles = get_title_by_features(
        genre,
        language,
        years,
        movies
    )

    if isinstance(encoded_titles, str):
        st.error(encoded_titles)  # Show error if no titles found
    else:
        # Randomly select one encoded title from the matches
        selected_movie_title = random.choice(encoded_titles)
        recommended_movies = recommend_movies(selected_movie_title, similarity_matrix, movie_titles, top_n=10)

        if recommended_movies:
            # Retrieve and display details for the selected movie
            columns_to_retrieve = ['Title', 'Overview', 'Release_Year', 'Genre', 'Vote_Average', 'Poster_Url']
            movie_info = get_specific_movie_details_by_title(selected_movie_title, movies, columns_to_retrieve)

            st.image(movie_info['Poster_Url'])
            st.subheader(f"🎬 {movie_info['Title']}")
            st.write(f"Overview: {movie_info['Overview']}")
            st.write(f"Average Rating: {movie_info['Vote_Average']} ⭐")
            st.write(f"Release Year: {movie_info['Release_Year']}")
            st.write(f"Genre: {movie_info['Genre']}")
            st.header("Want another movie? Press the button again!")
        
        else:
            st.write("No movies found matching the criteria.")

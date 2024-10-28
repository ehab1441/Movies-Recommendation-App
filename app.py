import streamlit as st
import pandas as pd
import random
from Recommendation_System import (
    recommend_movies,
    get_encoded_title_by_features,
    get_specific_movie_details_by_encoded_title,
)

# Load the movies data
movies = pd.read_csv('movies_data.csv', lineterminator='\n')

st.set_page_config(
    page_title="Movie Recommendation App",
    page_icon="🎬",
)

st.title("Movie Recommendation App")
st.write("Get recommendations based on your preferences!")

# Selection inputs for genre, language, and era
genre = st.selectbox("Select a genre", movies['Primary_Genre'].unique())
language = st.selectbox("Select a language", movies['Original_Language_Full'].unique())
era_input = st.selectbox("Select Era", sorted(movies['Release_Era'].unique()))

# Function to get the range of years for the selected era
def get_era_years(era_input):
    if not isinstance(era_input, int):
        raise ValueError("Invalid era input. Please provide an integer.")

    era_start = era_input
    era_end = era_start + 9

    return list(range(era_start, era_end + 1))

# Get the decade years from the selected era
years = get_era_years(era_input)

# Button to trigger movie recommendations
if st.sidebar.button("Recommend a movie"):
    # Get encoded titles based on user inputs
    encoded_titles = get_encoded_title_by_features(
        genre,
        language,
        years,  # Pass the list of years
        movies
    )

    if isinstance(encoded_titles, str):
        st.error(encoded_titles)
    else:
        selected_movie_title = random.choice(encoded_titles)
        recommended_movies = recommend_movies(selected_movie_title, similarity_matrix, movie_titles, top_n=10)

        if recommended_movies:
            # Get movie info for the selected movie title
            movie_info = get_specific_movie_details_by_encoded_title(selected_movie_title, movies)

            st.image(movie_info['Poster_Url'])
            st.subheader(f"🎬 {movie_info['Title']}")
            st.write(f"Overview: {movie_info['Overview']}")
            st.write(f"Average Rating: {movie_info['Vote_Average']} ⭐")
            st.write(f"Release Year: {movie_info['Release_Year']}")
            st.write(f"Genre: {movie_info['Genre']}")

            st.write("Recommended Movies:")
            for movie in recommended_movies:
                st.write(movie)
        else:
            st.write("No recommendations available.")

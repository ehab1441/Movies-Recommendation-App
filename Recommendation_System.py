import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import random
from Cleaning import pipeline

# Load the movies data (correcting the path)
movies = pd.read_csv("movies.csv", lineterminator='\n')
movies = pipeline(movies)
# Preparing data for similarity calculation
X = movies[['Genre_Encoded', 'Language_Encoded', 'Release_Year', 'Vote_Count', 'Vote_Average', 'Popularity']]
similarity_matrix = cosine_similarity(X)
movie_titles = movies['Title'].values

def recommend_movies(movie_title, similarity_matrix, movie_titles, top_n=5):
    """
    Recommends top_n movies similar to the given movie title based on cosine similarity.
    
    Parameters:
    - movie_title_encoded: Encoded title of the movie to find recommendations for.
    - similarity_matrix: Matrix containing cosine similarity scores between movies.
    - movie_titles: List of movie titles in the dataset.
    - top_n: Number of top recommendations to return.
    
    Returns:
    - List of recommended movie titles.
    """
    # Find the index of the encoded movie title
    movie_indices = np.where(movie_titles == movie_title)[0]
    
    if len(movie_indices) == 0:
        print(f"Encoded title '{movie_title}' not found in movie titles.")
        return []
    
    # Retrieve similarity scores for the target movie and sort to find top matches
    movie_idx = movie_indices[0]
    similar_movies = sorted(
        list(enumerate(similarity_matrix[movie_idx])),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Exclude the first item (itself) and get the top_n recommendations
    recommendations = [movie_titles[i[0]] for i in similar_movies[1:top_n+1]]
    return recommendations

def get_title_by_features(genre, language, release_years, movies_df):
    """
    Fetches movies by matching Genre, Language, and Release Year.
    
    Parameters:
    - genre: The genre to filter by.
    - language: The language to filter by.
    - release_years: List of release years to filter by.
    - movies_df: DataFrame containing the movies dataset.
    
    Returns:
    - Array of titles that match the specified feature values or a message if none are found.
    """
    # Filter movies based on genre, language, and release year criteria
    filtered_movies = movies_df[
        (movies_df['Primary_Genre'] == genre) &
        (movies_df['Original_Language_Full'] == language) &
        (movies_df['Release_Year'].isin(release_years))
    ]
    
    # Return titles or an error message if none match
    return filtered_movies['Title'].values if not filtered_movies.empty else "No movie found with the specified feature values."

# Example usage: defining criteria for recommendations
genre_input = 'Action'
language_input = 'English'
release_year_input = [2019, 2020]

# Get encoded titles based on selected features
titles = get_title_by_features(genre_input, language_input, release_year_input, movies)

if isinstance(titles, np.ndarray) and titles.size > 0:
    # Randomly select one title for further recommendation
    selected_title = random.choice(titles)
    print("Selected Title:", selected_title)

    # Get recommendations based on the selected title
    recommended_movies = recommend_movies(selected_title, similarity_matrix, movie_titles, top_n=10)
    print("Recommended Movies:", recommended_movies)

    def get_specific_movie_details_by_title(title, movies_df, columns):
        """
        Retrieves specific details for a movie by its encoded title.
        
        Parameters:
        - title: The encoded title of the movie to fetch details for.
        - movies_df: DataFrame containing the movies dataset.
        - columns: List of columns to retrieve for the movie details.
        
        Returns:
        - Series of specific movie details or None if the movie is not found.
        """
        # Filter the DataFrame to get the row corresponding to the encoded title
        movie_details = movies_df[movies_df['Title'] == title]
        return movie_details[columns].iloc[0] if not movie_details.empty else None
    
    seen_movies_file = 'seen_movies.txt'

    try:
        # Read previously seen movies from the file
        with open(seen_movies_file, 'r') as file:
            seen_movies = set(file.read().splitlines())
    except FileNotFoundError:
        # Initialize as an empty set if file does not exist
        seen_movies = set()

    # Filter out movies already seen from the recommendations
    unseen_movies = [movie for movie in recommended_movies if movie not in seen_movies]

    # Check if there are unseen movies left
    if unseen_movies:
        # Randomly select one unseen movie
        selected_movie = random.choice(unseen_movies)

        # Define columns to retrieve movie details
        columns_to_retrieve = ['Title', 'Overview', 'Release_Year', 'Genre', 'Vote_Average', 'Poster_Url']
        
        # Get the movie details
        movie_info = get_specific_movie_details_by_title(selected_movie, movies, columns_to_retrieve)

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

            # Mark this movie as seen by adding it to the set
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

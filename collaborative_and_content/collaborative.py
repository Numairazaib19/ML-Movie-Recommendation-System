import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import urllib.parse

# Set Streamlit page configuration
st.set_page_config(page_title="🎬 Movie Recommendation System", layout="wide")

# Load Data
@st.cache_data
def load_data():
    movies_df = pd.read_csv('movies.csv', usecols=['movieId', 'title'], dtype={'movieId': 'int32', 'title': 'str'})
    rating_df = pd.read_csv('ratings.csv', usecols=['userId', 'movieId', 'rating'], dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
    return movies_df, rating_df

movies_df, rating_df = load_data()

# Data Preparation
df = pd.merge(rating_df, movies_df, on='movieId')
combine_movie_rating = df.dropna(axis=0, subset=['title'])
movie_ratingCount = (combine_movie_rating.groupby(by=['title'])['rating']
                     .count().reset_index().rename(columns={'rating': 'totalRatingCount'})[['title', 'totalRatingCount']])

rating_with_totalRatingCount = combine_movie_rating.merge(movie_ratingCount, left_on='title', right_on='title', how='left')
popularity_threshold = 50
rating_popular_movie = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')

movie_features_df = rating_popular_movie.pivot_table(index='title', columns='userId', values='rating').fillna(0)
movie_features_df_matrix = csr_matrix(movie_features_df.values)

# Model Training
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(movie_features_df_matrix)

# Streamlit UI Styling (with background image and improved UI)
st.markdown(
    """
    <style>
    /* Title Styling */
    .title {
        text-align: center;
        font-size: 50px;
        color: #000000;
        font-weight: bold;
        margin-top: 50px;
        margin-bottom: 30px;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Center the description text */
    .description {
        text-align: center;
        font-size: 18px;
        color: #333333;
    }

    /* Recommendation Card Styling */
    .recommendation-container {
        background-color: #ECECEC; /* Light gray background */
        background: linear-gradient(145deg, #f0f0f0, #d9d9d9); /* Subtle gradient */
        border-radius: 12px; /* Slightly rounder corners */
        padding: 20px; /* Added padding for spaciousness */
        margin-bottom: 20px; /* Increased margin to give space between cards */
    
    /* Add a soft shadow */
    box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1);
    
    /* Smooth transition for transform and shadow */
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    /* Hover effect */
    .recommendation-container:hover {
        transform: scale(1.05); /* Slightly enlarge the card on hover */
        box-shadow: 0px 6px 25px rgba(0, 0, 0, 0.2); /* Stronger shadow on hover */
    }

    /* Optional: Adding a subtle border on hover */
    .recommendation-container:hover {
        border: 1px solid #ccc; /* Soft border when hovered */
    }


    .recommendation-title {
        font-size: 20px;
        color: #000000;
        text-decoration: none;
    }

    .recommendation-distance {
        font-size: 14px;
        color: #001A38;
        font-style: italic;
    }

    /* Sidebar button styling */
    .sidebar .sidebar-content .element-container {
        background-color: #e0f7fa; /* Light cyan background for the buttons */
        color: #000000; /* Black text color */
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .sidebar .sidebar-content .element-container:hover {
        background-color: #b2ebf2; /* Slightly darker cyan on hover */
    }
    /* Sidebar styling - no color */
    .sidebar .sidebar-content {
        background-color: transparent;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and Description
st.markdown("<h1 class='title'>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<div class='description'>This app recommends movies based on user ratings using K-Nearest Neighbors</div>", unsafe_allow_html=True)
#st.write("This app recommends movies based on user ratings using K-Nearest Neighbors.")

# Sidebar for User Selection
st.sidebar.header("Select Movie for Recommendation")
selected_movie = st.sidebar.selectbox("Choose a movie:", movie_features_df.index)
num_recommendations = st.sidebar.slider("Number of recommendations:", min_value=1, max_value=10, value=5)

# Helper function to generate IMDb search URL
def generate_imdb_url(title):
    base_url = "https://www.imdb.com/find?q="
    query = urllib.parse.quote_plus(title)
    return base_url + query

# Display Recommendations
if st.sidebar.button("Get Recommendations"):
    st.subheader(f"Recommendations for **{selected_movie}**")

    # Get index and recommendations
    query_index = movie_features_df.index.get_loc(selected_movie)
    distances, indices = model_knn.kneighbors(
        movie_features_df.iloc[query_index, :].values.reshape(1, -1), n_neighbors=num_recommendations + 1
    )

    # Display each recommendation with formatted container
    for i in range(1, len(distances.flatten())):
        movie_title = movie_features_df.index[indices.flatten()[i]]
        imdb_url = generate_imdb_url(movie_title)  # Generate IMDb search URL

        # Recommendation display
        st.markdown(f"<div class='recommendation-container'>"
                    f"<div class='recommendation-title'>{i}: <a href='{imdb_url}' target='_blank'>{movie_title}</a></div>"
                    f"<div class='recommendation-distance'>Distance Score: {distances.flatten()[i]:.4f}</div>"
                    "</div>",
                    unsafe_allow_html=True)

    # Display additional information link for the selected movie
    st.write(f"[More about '{selected_movie}' on IMDb]({generate_imdb_url(selected_movie)})")

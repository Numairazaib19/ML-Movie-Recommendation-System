import streamlit as st
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import urllib.parse

# Set Streamlit page configuration
st.set_page_config(page_title="🎬 Movie Recommendation System", layout="wide")

# Load Data for Collaborative Filtering
@st.cache_data
def load_collaborative_data():
    movies_df = pd.read_csv('movies.csv', usecols=['movieId', 'title'], dtype={'movieId': 'int32', 'title': 'str'})
    rating_df = pd.read_csv('ratings.csv', usecols=['userId', 'movieId', 'rating'], dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
    return movies_df, rating_df

# Load Data for Content-Based Filtering
@st.cache_data
def load_content_based_data():
    movies_df = pd.read_csv('tmdb_5000_credits.csv', dtype={'id': 'int32', 'title': 'str', 'genres': 'str', 'keywords': 'str', 'overview': 'str'})
    credits_df = pd.read_csv('tmdb_5000_movies.csv', dtype={'movie_id': 'int32', 'title': 'str', 'cast': 'str', 'crew': 'str'})
    return movies_df, credits_df

# Collaborative Filtering - Data Preparation
movies_df_cf, rating_df_cf = load_collaborative_data()
df = pd.merge(rating_df_cf, movies_df_cf, on='movieId')
combine_movie_rating = df.dropna(axis=0, subset=['title'])
movie_ratingCount = (combine_movie_rating.groupby(by=['title'])['rating']
                     .count().reset_index().rename(columns={'rating': 'totalRatingCount'})[['title', 'totalRatingCount']])

rating_with_totalRatingCount = combine_movie_rating.merge(movie_ratingCount, left_on='title', right_on='title', how='left')
popularity_threshold = 50
rating_popular_movie = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')

movie_features_df = rating_popular_movie.pivot_table(index='title', columns='userId', values='rating').fillna(0)
movie_features_df_matrix = csr_matrix(movie_features_df.values)

model_knn_cf = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn_cf.fit(movie_features_df_matrix)

# Content-Based Filtering - Data Preparation
movies_df_cb, credits_df_cb = load_content_based_data()
merged_df_cb = pd.merge(movies_df_cb, credits_df_cb, left_on='title', right_on='title')
merged_df_cb = merged_df_cb.fillna('')
merged_df_cb['content'] = merged_df_cb['genres'] + ' ' + merged_df_cb['keywords'] + ' ' + merged_df_cb['overview'] + ' ' + merged_df_cb['cast'] + ' ' + merged_df_cb['crew']

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
movie_features_tfidf = tfidf_vectorizer.fit_transform(merged_df_cb['content'])

model_knn_cb = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn_cb.fit(movie_features_tfidf)

# Custom CSS for Styling
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
        background-color: #ECECEC;
        background: linear-gradient(145deg, #f0f0f0, #d9d9d9);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    /* Hover effect */
    .recommendation-container:hover {
        transform: scale(1.05);
        box-shadow: 0px 6px 25px rgba(0, 0, 0, 0.2);
    }

    /* Recommendation title styling */
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
    </style>
    """,
    unsafe_allow_html=True
)

# Select which recommendation system to use
st.markdown("<h1 class='title'>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<div class='description'>Choose a recommendation system:</div>", unsafe_allow_html=True)

option = st.selectbox("Select Recommendation System", ["Collaborative Filtering", "Content-Based Filtering"])

# Collaborative Filtering Recommendation
if option == "Collaborative Filtering":
    st.markdown("<div class='description'>This system recommends movies based on user ratings.</div>", unsafe_allow_html=True)

    st.sidebar.header("Select Movie for Recommendation")
    selected_movie_cf = st.sidebar.selectbox("Choose a movie:", movie_features_df.index)
    num_recommendations_cf = st.sidebar.slider("Number of recommendations:", min_value=1, max_value=10, value=5)

    def generate_imdb_url(title):
        base_url = "https://www.imdb.com/find?q="
        query = urllib.parse.quote_plus(title)
        return base_url + query

    if st.sidebar.button("Get Recommendations"):
        st.subheader(f"Recommendations for **{selected_movie_cf}**")

        query_index = movie_features_df.index.get_loc(selected_movie_cf)
        distances, indices = model_knn_cf.kneighbors(
            movie_features_df.iloc[query_index, :].values.reshape(1, -1), n_neighbors=num_recommendations_cf + 1
        )

        for i in range(1, len(distances.flatten())):
            movie_title = movie_features_df.index[indices.flatten()[i]]
            imdb_url = generate_imdb_url(movie_title)
            st.markdown(f"<div class='recommendation-container'>"
                        f"<div class='recommendation-title'>{i}: <a href='{imdb_url}' target='_blank'>{movie_title}</a></div>"
                        f"<div class='recommendation-distance'>Distance Score: {distances.flatten()[i]:.4f}</div>"
                        "</div>",
                        unsafe_allow_html=True)
            
        # Add IMDb link for the selected movie
        selected_movie_url = generate_imdb_url(selected_movie_cf)
        st.markdown(f"<div class='selected-movie-link' style='text-align: right;'>"
                    f"<div><a href='{selected_movie_url}' target='_blank'>More about {selected_movie_cf} on IMDb</a></div>"
                    "</div>",
                    unsafe_allow_html=True)
    

# Content-Based Filtering Recommendation
elif option == "Content-Based Filtering":
    st.markdown("<div class='description'>This system recommends movies based on content (genres, keywords, overview, cast, crew).</div>", unsafe_allow_html=True)

    st.sidebar.header("Select Movie for Recommendation")
    selected_movie_cb = st.sidebar.selectbox("Choose a movie:", merged_df_cb['title'].tolist())
    num_recommendations_cb = st.sidebar.slider("Number of recommendations:", min_value=1, max_value=10, value=5)
    
    def generate_imdb_url(title):
        base_url = "https://www.imdb.com/find?q="
        query = urllib.parse.quote_plus(title)
        return base_url + query

    if st.sidebar.button("Get Recommendations"):
        st.subheader(f"Recommendations for **{selected_movie_cb}**")

        movie_idx = merged_df_cb[merged_df_cb['title'] == selected_movie_cb].index[0]
        distances, indices = model_knn_cb.kneighbors(
            movie_features_tfidf[movie_idx], n_neighbors=num_recommendations_cb + 1
        )

        for i in range(1, len(distances.flatten())):
            movie_title = merged_df_cb['title'].iloc[indices.flatten()[i]]
            imdb_url = generate_imdb_url(movie_title)
            st.markdown(f"<div class='recommendation-container'>"
                        f"<div class='recommendation-title'>{i}: <a href='{imdb_url}' target='_blank'>{movie_title}</a></div>"
                        f"<div class='recommendation-distance'>Distance Score: {distances.flatten()[i]:.4f}</div>"
                        "</div>",
                        unsafe_allow_html=True)

        # Add IMDb link for the selected movie
        selected_movie_url = generate_imdb_url(selected_movie_cb)
        st.markdown(f"<div class='selected-movie-link' style='text-align: right;'>"
                    f"<div><a href='{selected_movie_url}' target='_blank'>More about {selected_movie_cb} on IMDb</a></div>"
                    "</div>",
                    unsafe_allow_html=True)
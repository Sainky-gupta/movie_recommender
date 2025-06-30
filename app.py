import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.read_csv('filter_data.csv')
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Vectorize tags
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['tags']).toarray()

# Compute similarity
similarity = cosine_similarity(vectors)

# Recommend function
def recommend(movie):
    movie = movie.lower()
    movie_index = df[df['title'].str.lower() == movie].index
    if len(movie_index) == 0:
        return []
    movie_index = movie_index[0]
    distances = list(enumerate(similarity[movie_index]))
    movies_list = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
    return [df.iloc[i[0]].title for i in movies_list]

# Streamlit UI
st.title("🎬 Movie Recommendation System")
st.write("Select a movie from the dropdown to get 5 similar movies.")

selected_movie = st.selectbox("Choose a movie", df['title'].values)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    if recommendations:
        st.subheader(f"Top 5 movies similar to **{selected_movie}**:")
        for movie in recommendations:
            link = f"https://www.google.com/search?q={movie.replace(' ', '+')}+movie"
            st.markdown(f"- [{movie}]({link})")
    else:
        st.write("Sorry, could not find similar movies.")

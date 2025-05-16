import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ----------------------------
# Step 1: Load and Prepare Data

# ----------------------------
movies_df = pd.read_csv(r'C:\Users\medha\movie_project\tmdb_5000_movies.csv')
credits_df = pd.read_csv(r'C:\Users\medha\movie_project\tmdb_5000_credits.csv')

credits_df['cast_count'] = credits_df['cast'].apply(lambda x: len(eval(x)) if pd.notna(x) else 0)
credits_df['crew_count'] = credits_df['crew'].apply(lambda x: len(eval(x)) if pd.notna(x) else 0)

df = pd.merge(movies_df, credits_df[['movie_id', 'cast_count', 'crew_count']], left_on='id', right_on='movie_id', how='inner')

df = df[df['revenue'] > 0]
df = df[df['budget'] > 0]

df['genre'] = df['genres'].apply(lambda x: [d['name'] for d in eval(x)] if pd.notna(x) else [])
genre_exploded = df.explode('genre')
genre_dummies = pd.get_dummies(genre_exploded['genre'])
genre_dummies_grouped = genre_dummies.groupby(genre_exploded.index).sum()
df = pd.concat([df, genre_dummies_grouped], axis=1)

df['budget'] = np.log1p(df['budget'])
df['revenue'] = np.log1p(df['revenue'])
df['popularity'] = np.log1p(df['popularity'])

# Features
genre_columns = list(genre_dummies.columns)
features = ['budget', 'popularity', 'cast_count', 'crew_count'] + genre_columns
X = df[features]
y = df['revenue']

# ----------------------------
# Step 2: Train Random Forest
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# ----------------------------
# Step 3: Streamlit App
# ----------------------------
st.set_page_config(page_title="Movie Revenue Predictor", layout="centered")
st.title("🎬 Movie Revenue Predictor")
st.write("Enter movie details below to predict box office revenue.")

# Input Fields
budget = st.number_input("Budget (in USD)", min_value=10_000, max_value=500_000_000, value=100_000_000)
popularity = st.slider("Popularity", min_value=0.0, max_value=100.0, value=50.0)
cast_count = st.slider("Cast Count", min_value=1, max_value=50, value=20)
crew_count = st.slider("Crew Count", min_value=1, max_value=50, value=15)
selected_genres = st.multiselect("Select Genres", genre_columns, default=["Action", "Science Fiction"])

# Prediction
if st.button("Predict Revenue"):
    input_data = {
        'budget': np.log1p(budget),
        'popularity': np.log1p(popularity),
        'cast_count': cast_count,
        'crew_count': crew_count,
    }
    for genre in genre_columns:
        input_data[genre] = 1 if genre in selected_genres else 0

    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    predicted_log_revenue = rf_model.predict(input_df)[0]
    predicted_revenue = np.expm1(predicted_log_revenue)

    st.success(f"💰 Predicted Revenue: **${predicted_revenue:,.2f}**")

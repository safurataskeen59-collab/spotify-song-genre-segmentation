
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Spotify Music Recommender",
    page_icon="🎵",
    layout="wide"
)


# ==========================================================
# 2. AUDIO FEATURES
# ==========================================================

FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]


# ==========================================================
# 3. LOAD AND CLEAN DATASET
# ==========================================================

@st.cache_data
def prepare_data():

    df = pd.read_csv("output/spotify_clustered.csv")

    # Fill missing song names and artist names.
    df["track_name"] = (
        df["track_name"]
        .fillna("Unknown Track")
        .astype(str)
        .str.strip()
    )

    df["track_artist"] = (
        df["track_artist"]
        .fillna("Unknown Artist")
        .astype(str)
        .str.strip()
    )

    # Convert audio features into numeric values.
    for feature in FEATURES:
        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

    df["cluster"] = pd.to_numeric(
        df["cluster"],
        errors="coerce"
    )

    # Remove records with missing audio features or cluster labels.
    df = df.dropna(
        subset=FEATURES + ["cluster"]
    ).copy()

    # Remove invalid numerical values.
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=FEATURES + ["cluster"]
    ).copy()

    # Remove duplicate Spotify track IDs.
    if "track_id" in df.columns:
        df = df.drop_duplicates(
            subset=["track_id"],
            keep="first"
        )

    # Create normalized keys for matching song titles and artists.
    df["_name_key"] = (
        df["track_name"]
        .str.casefold()
        .str.replace(r"\s+", " ", regex=True)
    )

    df["_artist_key"] = (
        df["track_artist"]
        .str.casefold()
        .str.replace(r"\s+", " ", regex=True)
    )

    # Identify rows with known song names and artists.
    known_song = (
        ~df["_name_key"].isin(
            ["", "unknown", "unknown track"]
        )
        &
        ~df["_artist_key"].isin(
            ["", "unknown", "unknown artist"]
        )
    )

    # Keep one record per known song title and artist.
    known = df[known_song].drop_duplicates(
        subset=["_name_key", "_artist_key"],
        keep="first"
    )

    # Preserve records whose song identity is incomplete.
    unknown = df[~known_song].copy()

    # Assign unique internal keys to incomplete records
    # so unrelated unknown tracks are not merged.
    for index in unknown.index:
        unknown.at[index, "_name_key"] = (
            f"unknown_track_{index}"
        )
        unknown.at[index, "_artist_key"] = (
            f"unknown_artist_{index}"
        )

    # Combine the records and reset the row index.
    df = (
        pd.concat([known, unknown])
        .sort_index()
        .reset_index(drop=True)
    )

    # Scale the audio features.
    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(
        df[FEATURES]
    )

    return df, scaled_features


try:
    df, X_scaled = prepare_data()

except FileNotFoundError:
    st.error(
        "Clustered dataset not found. "
        "Please run analysis.py and clustering.py first."
    )
    st.stop()

except Exception as error:
    st.error(f"Unable to load the dataset: {error}")
    st.stop()


if df.empty:
    st.error("No usable songs were found in the dataset.")
    st.stop()


# ==========================================================
# 4. CREATE SONG LABELS FOR THE DROPDOWN
# ==========================================================

base_labels = (
    df["track_name"]
    + " — "
    + df["track_artist"]
)

display_labels = base_labels.tolist()

# Add an ID to any remaining duplicate display labels.
duplicate_labels = base_labels.duplicated(
    keep=False
).to_numpy()

for position in np.flatnonzero(duplicate_labels):

    if "track_id" in df.columns:

        track_id = str(
            df.iloc[position]["track_id"]
        )

        suffix = track_id[:8]

    else:

        suffix = str(position + 1)

    display_labels[position] = (
        f"{base_labels.iloc[position]} "
        f"[{suffix}]"
    )


# ==========================================================
# 5. RECOMMENDATION FUNCTION
# ==========================================================

def get_recommendations(
    selected_position,
    number_of_recommendations=10
):

    selected_song = df.iloc[selected_position]

    selected_cluster = selected_song["cluster"]

    # Find songs belonging to the same cluster.
    cluster_positions = np.flatnonzero(
        df["cluster"].to_numpy() == selected_cluster
    )

    # Get the selected song's scaled audio features.
    selected_vector = X_scaled[
        selected_position
    ].reshape(1, -1)

    # Get audio features of songs in this cluster.
    cluster_vectors = X_scaled[
        cluster_positions
    ]

    # Calculate cosine similarity.
    similarity_scores = cosine_similarity(
        selected_vector,
        cluster_vectors
    )[0]

    # Create the recommendation table.
    recommendations = df.iloc[
        cluster_positions
    ].copy()

    recommendations["similarity_score"] = (
        similarity_scores
    )

    # Exclude the selected song itself.
    recommendations = recommendations[
        recommendations.index != selected_position
    ]

    # Exclude any other entry with the same song title
    # and artist as the selected song.
    recommendations = recommendations[
        ~(
            (recommendations["_name_key"]
             == selected_song["_name_key"])
            &
            (recommendations["_artist_key"]
             == selected_song["_artist_key"])
        )
    ]

    # Keep only one occurrence of each song and artist.
    recommendations = recommendations.drop_duplicates(
        subset=["_name_key", "_artist_key"],
        keep="first"
    )

    # Rank songs by their similarity score.
    recommendations = recommendations.sort_values(
        by="similarity_score",
        ascending=False
    )

    return recommendations.head(
        number_of_recommendations
    )


# ==========================================================
# 6. APPLICATION HEADER
# ==========================================================

st.title("🎵 Spotify Music Recommender")

st.subheader(
    "Discover songs similar to your favourite track"
)

st.write(
    """
    This application uses **K-Means clustering** and
    **cosine similarity** to recommend songs based on
    their audio characteristics.
    """
)


# ==========================================================
# 7. SIDEBAR
# ==========================================================

st.sidebar.title("About the Project")

st.sidebar.write(
    """
    **Project:** Spotify Song Genre Segmentation

    **Clustering:** K-Means

    **Recommendation:** Cosine Similarity

    **Audio features:**
    - Danceability
    - Energy
    - Loudness
    - Speechiness
    - Acousticness
    - Instrumentalness
    - Liveness
    - Valence
    - Tempo
    """
)

st.sidebar.metric(
    "Unique songs",
    f"{len(df):,}"
)

st.sidebar.metric(
    "Clusters",
    int(df["cluster"].nunique())
)


# ==========================================================
# 8. SONG SELECTION
# ==========================================================

st.subheader("Select a Song")

selected_position = st.selectbox(
    "Choose a song:",
    options=list(range(len(df))),
    format_func=lambda position: display_labels[position]
)

selected_song = df.iloc[selected_position]


# ==========================================================
# 9. DISPLAY SELECTED SONG
# ==========================================================

st.divider()

st.subheader("Selected Song")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Song",
        selected_song["track_name"]
    )

with col2:
    st.metric(
        "Artist",
        selected_song["track_artist"]
    )

with col3:
    st.metric(
        "Genre",
        str(selected_song["playlist_genre"]).upper()
    )

with col4:
    st.metric(
        "Cluster",
        int(selected_song["cluster"])
    )


# ==========================================================
# 10. AUDIO CHARACTERISTICS
# ==========================================================

st.subheader("Audio Characteristics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Danceability",
        f"{selected_song['danceability']:.2f}"
    )

with col2:
    st.metric(
        "Energy",
        f"{selected_song['energy']:.2f}"
    )

with col3:
    st.metric(
        "Valence",
        f"{selected_song['valence']:.2f}"
    )

with col4:
    st.metric(
        "Tempo",
        f"{selected_song['tempo']:.1f} BPM"
    )


# ==========================================================
# 11. GENERATE RECOMMENDATIONS
# ==========================================================

st.divider()

number_of_recommendations = st.selectbox(
    "Number of recommendations:",
    options=[5, 10, 15],
    index=1
)

if st.button(
    "🎧 Recommend Similar Songs",
    use_container_width=True
):

    recommendations = get_recommendations(
        selected_position,
        number_of_recommendations
    )

    st.subheader("🎶 Recommended Songs")

    if recommendations.empty:

        st.warning(
            "No other unique songs were found in this cluster."
        )

    else:

        for number, (_, song) in enumerate(
            recommendations.iterrows(),
            start=1
        ):

            st.markdown(
                f"**{number}. {song['track_name']}** "
                f"— {song['track_artist']}"
            )

            st.caption(
                f"Genre: {str(song['playlist_genre']).upper()} "
                f"| Cluster: {int(song['cluster'])} "
                f"| Cosine similarity score: "
                f"{song['similarity_score']:.4f}"
            )

            st.divider()

        st.caption(
            "Similarity scores compare the selected audio features. "
            "They are not probabilities or guaranteed percentages."
        )


# ==========================================================
# 12. HOW THE SYSTEM WORKS
# ==========================================================

with st.expander("How does the recommendation system work?"):

    st.write(
        """
        1. The dataset is cleaned to reduce duplicate songs.
        2. K-Means groups songs using their audio characteristics.
        3. The system identifies the selected song's cluster.
        4. Cosine similarity compares songs within that cluster.
        5. The system displays the most similar distinct tracks.
        """
    )

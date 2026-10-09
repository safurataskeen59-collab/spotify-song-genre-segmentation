import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================================
# SPOTIFY MUSIC RECOMMENDATION SYSTEM
# ==========================================================

print("\n" + "=" * 70)
print("SPOTIFY MUSIC RECOMMENDATION SYSTEM")
print("=" * 70)


# ==========================================================
# 1. LOAD CLUSTERED DATASET
# ==========================================================

file_path = "output/spotify_clustered.csv"

df = pd.read_csv(file_path)

print("\nDataset loaded successfully.")

print(
    f"Total songs: {len(df):,}"
)

print(
    f"Total clusters: {df['cluster'].nunique()}"
)


# ==========================================================
# 2. SELECT FEATURES FOR SIMILARITY
# ==========================================================

features = [
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
# 3. SCALE FEATURES
# ==========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    df[features]
)

print("\nAudio features scaled successfully.")


# ==========================================================
# 4. CREATE SONG SEARCH FUNCTION
# ==========================================================

def search_songs(song_name, number_of_results=10):

    results = df[
        df["track_name"]
        .str.contains(
            song_name,
            case=False,
            na=False
        )
    ]

    return results.head(number_of_results)


# ==========================================================
# 5. RECOMMENDATION FUNCTION
# ==========================================================

def recommend_songs(
    song_name,
    number_of_recommendations=10
):

    # ------------------------------------------------------
    # Find the selected song
    # ------------------------------------------------------

    matches = df[
        df["track_name"]
        .str.contains(
            song_name,
            case=False,
            na=False
        )
    ]

    if len(matches) == 0:

        print(
            "\nNo song found with that name."
        )

        return None


    # Take the first matching song
    selected_index = matches.index[0]

    selected_song = df.loc[
        selected_index
    ]

    selected_cluster = selected_song[
        "cluster"
    ]


    print("\n" + "=" * 70)
    print("SELECTED SONG")
    print("=" * 70)

    print(
        f"Song: {selected_song['track_name']}"
    )

    print(
        f"Artist: {selected_song['track_artist']}"
    )

    print(
        f"Genre: {selected_song['playlist_genre']}"
    )

    print(
        f"Cluster: {selected_cluster}"
    )


    # ------------------------------------------------------
    # Get songs from the same cluster
    # ------------------------------------------------------

    cluster_indices = df[
        df["cluster"] == selected_cluster
    ].index


    # ------------------------------------------------------
    # Calculate similarity
    # ------------------------------------------------------

    selected_vector = X_scaled[
        df.index.get_loc(selected_index)
    ].reshape(1, -1)


    cluster_vectors = X_scaled[
        [
            df.index.get_loc(i)
            for i in cluster_indices
        ]
    ]


    similarity_scores = cosine_similarity(
        selected_vector,
        cluster_vectors
    )[0]


    # ------------------------------------------------------
    # Create recommendation dataframe
    # ------------------------------------------------------

    recommendations = df.loc[
        cluster_indices
    ].copy()

    recommendations[
        "similarity"
    ] = similarity_scores


    # ------------------------------------------------------
    # Remove selected song
    # ------------------------------------------------------

    recommendations = recommendations[
        recommendations.index != selected_index
    ]


    # ------------------------------------------------------
    # Sort by similarity
    # ------------------------------------------------------

    recommendations = (
        recommendations
        .sort_values(
            by="similarity",
            ascending=False
        )
        .head(
            number_of_recommendations
        )
    )


    # ======================================================
    # DISPLAY RECOMMENDATIONS
    # ======================================================

    print("\n" + "=" * 70)
    print("RECOMMENDED SONGS")
    print("=" * 70)


    display_columns = [
        "track_name",
        "track_artist",
        "playlist_genre",
        "cluster",
        "similarity"
    ]


    result = recommendations[
        display_columns
    ].copy()


    result["similarity"] = (
        result["similarity"]
        .round(4)
    )


    print(
        result.to_string(
            index=False
        )
    )


    return result


# ==========================================================
# 6. TEST THE RECOMMENDATION SYSTEM
# ==========================================================

print("\nExample recommendation test:")

# Take a known song from the dataset
example_song = df.iloc[0]["track_name"]

print(
    f"\nSearching recommendations for: "
    f"{example_song}"
)

recommend_songs(
    example_song,
    number_of_recommendations=10
)


# ==========================================================
# 7. INTERACTIVE MODE
# ==========================================================

print("\n" + "=" * 70)
print("INTERACTIVE SONG RECOMMENDER")
print("=" * 70)

print(
    "\nEnter a song name to get recommendations."
)

print(
    "Type 'exit' to stop."
)


while True:

    user_input = input(
        "\nEnter song name: "
    ).strip()


    if user_input.lower() == "exit":

        print(
            "\nRecommendation system closed."
        )

        break


    if user_input == "":
        print(
            "Please enter a song name."
        )
        continue


    recommend_songs(
        user_input,
        number_of_recommendations=10
    )
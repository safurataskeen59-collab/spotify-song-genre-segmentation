import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# SPOTIFY SONG GENRE SEGMENTATION PROJECT
# ==========================================================

# ----------------------------------------------------------
# 1. CREATE OUTPUT FOLDER
# ----------------------------------------------------------

os.makedirs("output", exist_ok=True)

# ----------------------------------------------------------
# 2. LOAD DATASET
# ----------------------------------------------------------

file_path = "dataset/spotify dataset.csv"

df = pd.read_csv(file_path)

print("\n" + "=" * 70)
print("SPOTIFY SONG GENRE SEGMENTATION PROJECT")
print("=" * 70)

print(f"\nOriginal dataset shape: {df.shape}")


# ----------------------------------------------------------
# 3. BASIC DATASET INFORMATION
# ----------------------------------------------------------

print("\n--- COLUMN NAMES ---")
for column in df.columns:
    print(column)


print("\n--- DATA TYPES ---")
print(df.dtypes)


# ----------------------------------------------------------
# 4. MISSING VALUE ANALYSIS
# ----------------------------------------------------------

print("\n--- MISSING VALUES BEFORE CLEANING ---")

missing = df.isnull().sum()
missing = missing[missing > 0]

if len(missing) == 0:
    print("No missing values found.")
else:
    print(missing)


# ----------------------------------------------------------
# 5. HANDLE MISSING VALUES
# ----------------------------------------------------------

text_columns = [
    "track_name",
    "track_artist",
    "track_album_name"
]

for column in text_columns:
    df[column] = df[column].fillna("Unknown")


print("\nMissing values after cleaning:")

total_missing = df.isnull().sum().sum()

if total_missing == 0:
    print("No missing values remain.")
else:
    print(df.isnull().sum()[df.isnull().sum() > 0])


# ----------------------------------------------------------
# 6. DUPLICATE ANALYSIS
# ----------------------------------------------------------

duplicate_rows = df.duplicated().sum()
duplicate_tracks = df["track_id"].duplicated().sum()

print("\n--- DUPLICATE ANALYSIS ---")
print(f"Completely duplicated rows: {duplicate_rows}")
print(f"Repeated track IDs: {duplicate_tracks}")

# We do NOT remove repeated track IDs because the
# same song can appear in different playlists.


# ----------------------------------------------------------
# 7. NUMERICAL STATISTICS
# ----------------------------------------------------------

print("\n--- NUMERICAL STATISTICS ---")

numeric_summary = df.describe()

print(numeric_summary)

numeric_summary.to_csv("output/numerical_summary.csv")


# ----------------------------------------------------------
# 8. GENRE ANALYSIS
# ----------------------------------------------------------

genre_counts = df["playlist_genre"].value_counts()

print("\n--- PLAYLIST GENRE DISTRIBUTION ---")
print(genre_counts)


# ----------------------------------------------------------
# 9. SUBGENRE ANALYSIS
# ----------------------------------------------------------

subgenre_counts = df["playlist_subgenre"].value_counts()

print("\n--- TOP 15 PLAYLIST SUBGENRES ---")
print(subgenre_counts.head(15))


# ----------------------------------------------------------
# 10. TOP ARTISTS
# ----------------------------------------------------------

top_artists = df["track_artist"].value_counts().head(15)

print("\n--- TOP 15 ARTISTS ---")
print(top_artists)


# ----------------------------------------------------------
# 11. MOST POPULAR SONGS
# ----------------------------------------------------------

popular_songs = (
    df[
        [
            "track_name",
            "track_artist",
            "track_popularity"
        ]
    ]
    .sort_values(
        by="track_popularity",
        ascending=False
    )
    .head(15)
)

print("\n--- TOP 15 MOST POPULAR SONGS ---")
print(popular_songs.to_string(index=False))


# ----------------------------------------------------------
# 12. AUDIO FEATURES
# ----------------------------------------------------------

audio_features = [
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


# ----------------------------------------------------------
# 13. AUDIO FEATURE SUMMARY
# ----------------------------------------------------------

audio_summary = df[audio_features].describe().T

print("\n--- AUDIO FEATURE SUMMARY ---")
print(audio_summary)

audio_summary.to_csv("output/audio_feature_summary.csv")


# ==========================================================
# VISUALIZATION SECTION
# ==========================================================


# ----------------------------------------------------------
# 14. GENRE DISTRIBUTION PLOT
# ----------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="playlist_genre",
    order=genre_counts.index
)

plt.title("Distribution of Playlist Genres")
plt.xlabel("Playlist Genre")
plt.ylabel("Number of Songs")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(
    "output/genre_distribution.png",
    dpi=300
)

plt.show()


# ----------------------------------------------------------
# 15. SUBGENRE DISTRIBUTION PLOT
# ----------------------------------------------------------

plt.figure(figsize=(12, 8))

sns.countplot(
    data=df,
    y="playlist_subgenre",
    order=subgenre_counts.head(20).index
)

plt.title("Top 20 Playlist Subgenres")
plt.xlabel("Number of Songs")
plt.ylabel("Playlist Subgenre")
plt.tight_layout()

plt.savefig(
    "output/subgenre_distribution.png",
    dpi=300
)

plt.show()


# ----------------------------------------------------------
# 16. AUDIO FEATURE DISTRIBUTIONS
# ----------------------------------------------------------

for feature in audio_features:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        df[feature],
        bins=30,
        kde=True
    )

    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        f"output/{feature}_distribution.png",
        dpi=300
    )

    plt.show()


# ----------------------------------------------------------
# 17. BOXPLOTS FOR AUDIO FEATURES
# ----------------------------------------------------------

plt.figure(figsize=(14, 7))

sns.boxplot(
    data=df[audio_features]
)

plt.title("Boxplot of Spotify Audio Features")
plt.xticks(
    range(len(audio_features)),
    audio_features,
    rotation=45
)

plt.tight_layout()

plt.savefig(
    "output/audio_features_boxplot.png",
    dpi=300
)

plt.show()


# ----------------------------------------------------------
# 18. CORRELATION MATRIX
# ----------------------------------------------------------

correlation_features = [
    "track_popularity",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms"
]

correlation_matrix = df[
    correlation_features
].corr()

print("\n--- CORRELATION MATRIX ---")
print(correlation_matrix.round(2))


# Save correlation matrix
correlation_matrix.to_csv(
    "output/correlation_matrix.csv"
)


# ----------------------------------------------------------
# 19. CORRELATION HEATMAP
# ----------------------------------------------------------

plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Matrix of Spotify Features")
plt.tight_layout()

plt.savefig(
    "output/correlation_heatmap.png",
    dpi=300
)

plt.show()


# ----------------------------------------------------------
# 20. AVERAGE AUDIO FEATURES BY GENRE
# ----------------------------------------------------------

genre_audio_means = (
    df.groupby("playlist_genre")[audio_features]
    .mean()
)

print("\n--- AVERAGE AUDIO FEATURES BY GENRE ---")
print(genre_audio_means.round(3))

genre_audio_means.to_csv(
    "output/genre_audio_means.csv"
)


# ----------------------------------------------------------
# 21. ENERGY BY GENRE
# ----------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.barplot(
    data=genre_audio_means.reset_index(),
    x="playlist_genre",
    y="energy"
)

plt.title("Average Energy by Playlist Genre")
plt.xlabel("Playlist Genre")
plt.ylabel("Average Energy")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(
    "output/average_energy_by_genre.png",
    dpi=300
)

plt.show()


# ----------------------------------------------------------
# 22. DANCEABILITY BY GENRE
# ----------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.barplot(
    data=genre_audio_means.reset_index(),
    x="playlist_genre",
    y="danceability"
)

plt.title("Average Danceability by Playlist Genre")
plt.xlabel("Playlist Genre")
plt.ylabel("Average Danceability")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(
    "output/average_danceability_by_genre.png",
    dpi=300
)

plt.show()


# ----------------------------------------------------------
# 23. VALENCE BY GENRE
# ----------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.barplot(
    data=genre_audio_means.reset_index(),
    x="playlist_genre",
    y="valence"
)

plt.title("Average Valence by Playlist Genre")
plt.xlabel("Playlist Genre")
plt.ylabel("Average Valence")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(
    "output/average_valence_by_genre.png",
    dpi=300
)

plt.show()


# ----------------------------------------------------------
# 24. FINAL CLEANED DATASET
# ----------------------------------------------------------

df.to_csv(
    "output/spotify_cleaned.csv",
    index=False
)


# ----------------------------------------------------------
# FINAL MESSAGE
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("DATA PREPROCESSING AND ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated files are available inside the 'output' folder.")
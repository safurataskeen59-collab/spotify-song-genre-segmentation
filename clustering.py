import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# ==========================================================
# SPOTIFY SONG GENRE SEGMENTATION
# K-MEANS CLUSTERING
# ==========================================================

print("\n" + "=" * 70)
print("SPOTIFY SONG GENRE SEGMENTATION - K-MEANS CLUSTERING")
print("=" * 70)


# ==========================================================
# 1. CREATE OUTPUT FOLDER
# ==========================================================

os.makedirs("output", exist_ok=True)


# ==========================================================
# 2. LOAD CLEANED DATASET
# ==========================================================

file_path = "output/spotify_cleaned.csv"

if not os.path.exists(file_path):
    print("\nERROR: Cleaned dataset not found!")
    print("Please run analysis.py first.")
    exit()

df = pd.read_csv(file_path)

print("\nDataset loaded successfully.")
print("Dataset shape:", df.shape)


# ==========================================================
# 3. SELECT AUDIO FEATURES
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

print("\nFeatures used for clustering:")

for feature in features:
    print("-", feature)


# ==========================================================
# 4. CREATE FEATURE DATA
# ==========================================================

X = df[features].copy()

print("\nFeature data shape:", X.shape)


# ==========================================================
# 5. FEATURE SCALING
# ==========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nFeature scaling completed successfully.")


# ==========================================================
# 6. ELBOW METHOD
# ==========================================================

print("\n" + "=" * 70)
print("CALCULATING ELBOW METHOD")
print("=" * 70)

inertia_values = []

k_values = range(2, 11)

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    # IMPORTANT:
    # Correct KMeans attribute is inertia_
    inertia_values.append(kmeans.inertia_)

    print(
        f"K = {k} | Inertia = {kmeans.inertia_:.2f}"
    )


# ==========================================================
# 7. PLOT ELBOW METHOD
# ==========================================================

plt.figure(figsize=(10, 6))

plt.plot(
    k_values,
    inertia_values,
    marker="o"
)

plt.title("Elbow Method for Optimal Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(list(k_values))
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "output/elbow_method.png",
    dpi=300
)

plt.show()


print("\nElbow Method graph saved successfully.")


# ==========================================================
# 8. SILHOUETTE SCORE
# ==========================================================

print("\n" + "=" * 70)
print("CALCULATING SILHOUETTE SCORES")
print("=" * 70)

silhouette_values = []

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        cluster_labels
    )

    silhouette_values.append(score)

    print(
        f"K = {k} | Silhouette Score = {score:.4f}"
    )


# ==========================================================
# 9. PLOT SILHOUETTE SCORES
# ==========================================================

plt.figure(figsize=(10, 6))

plt.plot(
    k_values,
    silhouette_values,
    marker="o"
)

plt.title("Silhouette Score for Different Values of K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(list(k_values))
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "output/silhouette_scores.png",
    dpi=300
)

plt.show()


print("\nSilhouette Score graph saved successfully.")


# ==========================================================
# 10. FIND BEST K
# ==========================================================

best_k = k_values[
    np.argmax(silhouette_values)
]

best_score = max(silhouette_values)

print("\n" + "=" * 70)
print("BEST CLUSTER NUMBER")
print("=" * 70)

print(f"Best K based on Silhouette Score: {best_k}")
print(f"Best Silhouette Score: {best_score:.4f}")


# ==========================================================
# 11. TRAIN FINAL K-MEANS MODEL
# ==========================================================

print("\n" + "=" * 70)
print("TRAINING FINAL K-MEANS MODEL")
print("=" * 70)

final_model = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["cluster"] = final_model.fit_predict(X_scaled)

print("\nFinal K-Means model trained successfully.")


# ==========================================================
# 12. CLUSTER DISTRIBUTION
# ==========================================================

cluster_counts = (
    df["cluster"]
    .value_counts()
    .sort_index()
)

print("\n" + "=" * 70)
print("SONGS IN EACH CLUSTER")
print("=" * 70)

print(cluster_counts)


# Save cluster counts
cluster_counts.to_csv(
    "output/cluster_counts.csv"
)


# ==========================================================
# 13. PCA DIMENSIONALITY REDUCTION
# ==========================================================

print("\n" + "=" * 70)
print("PERFORMING PCA")
print("=" * 70)

pca = PCA(
    n_components=2,
    random_state=42
)

X_pca = pca.fit_transform(X_scaled)

df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

explained_variance = pca.explained_variance_ratio_

print(
    f"\nPC1 explains {explained_variance[0] * 100:.2f}% of variance."
)

print(
    f"PC2 explains {explained_variance[1] * 100:.2f}% of variance."
)

print(
    f"Total explained variance: "
    f"{sum(explained_variance) * 100:.2f}%"
)


# ==========================================================
# 14. PCA CLUSTER VISUALIZATION
# ==========================================================

plt.figure(figsize=(11, 7))

scatter = plt.scatter(
    df["PCA1"],
    df["PCA2"],
    c=df["cluster"],
    cmap="viridis",
    alpha=0.5,
    s=12
)

plt.title(
    "Spotify Songs Clusters using K-Means"
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.colorbar(
    scatter,
    label="Cluster"
)

plt.tight_layout()

plt.savefig(
    "output/pca_clusters.png",
    dpi=300
)

plt.show()


print("\nPCA cluster visualization saved successfully.")


# ==========================================================
# 15. AVERAGE AUDIO FEATURES FOR EACH CLUSTER
# ==========================================================

cluster_means = (
    df.groupby("cluster")[features]
    .mean()
)

print("\n" + "=" * 70)
print("AVERAGE AUDIO FEATURES FOR EACH CLUSTER")
print("=" * 70)

print(
    cluster_means.round(3)
)


cluster_means.to_csv(
    "output/cluster_audio_means.csv"
)


# ==========================================================
# 16. CLUSTER VS PLAYLIST GENRE
# ==========================================================

cluster_genre = pd.crosstab(
    df["cluster"],
    df["playlist_genre"]
)

print("\n" + "=" * 70)
print("CLUSTER VS PLAYLIST GENRE")
print("=" * 70)

print(cluster_genre)


cluster_genre.to_csv(
    "output/cluster_vs_genre.csv"
)


# ==========================================================
# 17. CLUSTER VS PLAYLIST GENRE HEATMAP
# ==========================================================

plt.figure(figsize=(10, 6))

sns.heatmap(
    cluster_genre,
    annot=True,
    fmt="d",
    cmap="Blues",
    linewidths=0.5
)

plt.title(
    "Relationship Between K-Means Clusters and Playlist Genres"
)

plt.xlabel("Playlist Genre")
plt.ylabel("Cluster")

plt.tight_layout()

plt.savefig(
    "output/cluster_vs_genre_heatmap.png",
    dpi=300
)

plt.show()


print(
    "\nCluster vs Genre heatmap saved successfully."
)


# ==========================================================
# 18. DOMINANT GENRE OF EACH CLUSTER
# ==========================================================

dominant_genres = cluster_genre.idxmax(
    axis=1
)

print("\n" + "=" * 70)
print("DOMINANT GENRE FOR EACH CLUSTER")
print("=" * 70)

for cluster, genre in dominant_genres.items():

    print(
        f"Cluster {cluster} -> {genre}"
    )


# ==========================================================
# 19. CLUSTER VS PLAYLIST NAME
# ==========================================================

cluster_playlist = pd.crosstab(
    df["cluster"],
    df["playlist_name"]
)

cluster_playlist.to_csv(
    "output/cluster_vs_playlist_name.csv"
)

print(
    "\nCluster vs Playlist Name analysis completed."
)


# ==========================================================
# 20. FIND TOP PLAYLISTS FOR EACH CLUSTER
# ==========================================================

print("\n" + "=" * 70)
print("TOP PLAYLISTS FOR EACH CLUSTER")
print("=" * 70)

for cluster in sorted(df["cluster"].unique()):

    cluster_data = df[
        df["cluster"] == cluster
    ]

    top_playlists = (
        cluster_data["playlist_name"]
        .value_counts()
        .head(5)
    )

    print(
        f"\nCluster {cluster}:"
    )

    for playlist, count in top_playlists.items():

        print(
            f"  {playlist} -> {count} songs"
        )


# ==========================================================
# 21. FIND TOP SONGS IN EACH CLUSTER
# ==========================================================

print("\n" + "=" * 70)
print("TOP POPULAR SONGS IN EACH CLUSTER")
print("=" * 70)

for cluster in sorted(df["cluster"].unique()):

    cluster_songs = (
        df[df["cluster"] == cluster][
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
        .head(5)
    )

    print(
        f"\nCluster {cluster}:"
    )

    print(
        cluster_songs.to_string(
            index=False
        )
    )


# ==========================================================
# 22. CREATE CLUSTER LABEL
# ==========================================================

df["cluster_label"] = (
    "Cluster "
    + df["cluster"].astype(str)
)


# ==========================================================
# 23. SAVE FINAL CLUSTERED DATASET
# ==========================================================

output_file = "output/spotify_clustered.csv"

df.to_csv(
    output_file,
    index=False
)

print(
    f"\nFinal clustered dataset saved to: {output_file}"
)


# ==========================================================
# 24. FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 70)
print("K-MEANS CLUSTERING COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"\nTotal songs analysed: {len(df):,}")

print(
    f"Number of clusters created: {best_k}"
)

print(
    f"Silhouette Score: {best_score:.4f}"
)

print("\nOutput files created:")

output_files = [
    "elbow_method.png",
    "silhouette_scores.png",
    "pca_clusters.png",
    "cluster_counts.csv",
    "cluster_audio_means.csv",
    "cluster_vs_genre.csv",
    "cluster_vs_genre_heatmap.png",
    "cluster_vs_playlist_name.csv",
    "spotify_clustered.csv"
]

for file in output_files:
    print(f"  - output/{file}")

print("\nProject clustering stage completed!")
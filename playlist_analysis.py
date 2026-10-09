import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# SPOTIFY PLAYLIST ANALYSIS BY CLUSTER
# ==========================================================

df = pd.read_csv("output/spotify_clustered.csv")

print("=" * 70)
print("PLAYLIST ANALYSIS BY CLUSTER")
print("=" * 70)


# ----------------------------------------------------------
# 1. TOP PLAYLISTS IN EACH CLUSTER
# ----------------------------------------------------------

for cluster in sorted(df["cluster"].unique()):

    print("\n" + "-" * 60)
    print(f"CLUSTER {cluster}")
    print("-" * 60)

    cluster_data = df[df["cluster"] == cluster]

    top_playlists = (
        cluster_data["playlist_name"]
        .value_counts()
        .head(10)
    )

    print(top_playlists)


# ----------------------------------------------------------
# 2. TOP 10 PLAYLISTS OVERALL
# ----------------------------------------------------------

top_playlists_overall = (
    df["playlist_name"]
    .value_counts()
    .head(10)
)

print("\n" + "=" * 70)
print("TOP 10 PLAYLISTS OVERALL")
print("=" * 70)

print(top_playlists_overall)


# ----------------------------------------------------------
# 3. PLAYLIST vs CLUSTER PLOT
# ----------------------------------------------------------

top_playlist_names = (
    df["playlist_name"]
    .value_counts()
    .head(10)
    .index
)

playlist_cluster_data = df[
    df["playlist_name"].isin(top_playlist_names)
]

playlist_cluster_table = pd.crosstab(
    playlist_cluster_data["playlist_name"],
    playlist_cluster_data["cluster"]
)


# ----------------------------------------------------------
# 4. PLOT
# ----------------------------------------------------------

playlist_cluster_table.plot(
    kind="bar",
    figsize=(14, 7)
)

plt.title(
    "Top 10 Playlists Distribution Across Clusters"
)

plt.xlabel("Playlist Name")
plt.ylabel("Number of Songs")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    "output/top_playlists_by_cluster.png",
    dpi=300
)

plt.show()


print(
    "\nPlaylist analysis completed successfully."
)
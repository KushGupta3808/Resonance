"""
Resonance - Content-Based Recommender (Stage 1: brute-force baseline)

Represents every song as a feature vector (a fixed-order list of its
audio features), then finds the most similar songs to a given song
using cosine similarity - comparing the ANGLE between two songs'
vectors, not their raw distance, which matters because features like
tempo (roughly 50-200) and danceability (0-1) live on very different
numeric scales.

This is deliberately brute-force: to recommend for one song, we
compute its similarity against ALL other songs, then sort. Works fine
at 114k songs, but doesn't scale forever - Stage 2 swaps in FAISS and
we measure the actual speed difference, same "naive first" pattern as
GateKeeper's rate limiter.
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# The audio features we're using as the song's "coordinates." Chosen
# because they're all genuinely about how the song SOUNDS, not
# metadata like popularity or duration.
FEATURE_COLUMNS = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]


class ContentBasedRecommender:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.df = self.df.drop_duplicates(subset="track_name").reset_index(drop=True)

        # StandardScaler rescales each feature to have mean 0, std 1.
        # WHY THIS MATTERS: loudness ranges roughly -60 to 0, while
        # danceability ranges 0 to 1. Without rescaling, loudness would
        # dominate the similarity calculation purely because its raw
        # numbers are bigger, not because it's actually more important.
        self.scaler = StandardScaler()
        self.feature_matrix = self.scaler.fit_transform(self.df[FEATURE_COLUMNS])

        # track_name -> row index, so we can look up a song by name
        self.name_to_index = {
            name: idx for idx, name in enumerate(self.df["track_name"])
        }

    def recommend(self, track_name: str, top_n: int = 5) -> pd.DataFrame:
        if track_name not in self.name_to_index:
            raise ValueError(f"'{track_name}' not found in dataset")

        idx = self.name_to_index[track_name]
        query_vector = self.feature_matrix[idx].reshape(1, -1)

        # Compare the query song's vector against EVERY song's vector.
        # cosine_similarity here returns a 1 x N array of similarity
        # scores, one per song in the dataset.
        similarities = cosine_similarity(query_vector, self.feature_matrix)[0]

        # argsort gives indices that would sort ascending; [::-1] flips
        # to descending (most similar first). We skip index 0 of the
        # sorted result since that's always the song itself (perfect
        # similarity with itself).
        similar_indices = similarities.argsort()[::-1][1:top_n + 1]

        results = self.df.iloc[similar_indices][
            ["track_name", "artists", "track_genre"]
        ].copy()
        results["similarity"] = similarities[similar_indices]
        return results

# Resonance — Technical Requirements Document (TRD)

**Status:** Draft v1
**Last updated:** August 2026

---

## 1. Data Sourcing (read this first — it shapes everything else)

**Original plan vs. reality:** Spotify deprecated the `audio-features`, `audio-analysis`, `recommendations`, and `related-artists` endpoints for all new developer apps in November 2024, and further restricted Developer Mode in February 2026. Spotify's developer policy also explicitly prohibits using their data to train ML models. This means building against the live Spotify API is not viable for a new project - not a workaround-able limitation, a hard wall. This is documented here rather than hidden, because "I checked and the obvious approach doesn't work, so I used a real alternative" is a more honest and more interesting engineering story than pretending the live API was used.

**Content-based data:** `spotify-tracks-dataset-detailed.csv` — 114,000 real Spotify tracks with genuine audio features (danceability, energy, tempo, valence, acousticness, etc.), originally collected via Spotify's API before the endpoint lockdown, now distributed as a static, public, pre-collected dataset (a common and legitimate practice in ML research and education - using a frozen public dataset instead of a live API is standard when the live API isn't accessible).

**Collaborative filtering data:** the HetRec 2011 Last.fm dataset (`user_artists.dat` + `artists.dat`) — ~92,834 real user-artist listening-count interactions from 1,892 real (anonymized) users across 17,632 artists. This is a long-standing, widely-cited academic recommender-systems benchmark dataset, published for the 2nd International Workshop on Information Heterogeneity and Fusion in Recommender Systems (HetRec 2011). Note this is artist-level, not track-level, interaction data - a user's "weight" (listening count) is tied to an artist, not an individual song. This is an intentional, honest scope note: real per-track, per-user Spotify listening data isn't publicly available anywhere (that's private user data), so artist-level collaborative filtering, cross-referenced against the track-level content-based system by artist name, is the most legitimate combination of real datasets available.

## 2. Architecture Overview

```
                    ┌─────────────────────────┐
                    │   Content-Based Engine    │
                    │  (audio feature vectors)  │
                    │   cosine similarity /      │
                    │   FAISS vector search      │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │      Hybrid Scorer        │
                    │  blends both signals       │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▲────────────┐
                    │ Collaborative Filtering    │
                    │  (matrix factorization on  │
                    │   user-artist interactions)│
                    └─────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │      FastAPI service       │
                    │   /recommend endpoint      │
                    └─────────────────────────┘
```

## 3. Content-Based Filtering

**Analogy:** Every song becomes a point in space, where each audio feature (tempo, energy, danceability...) is one dimension of that point's location. Songs that "sound similar" end up as points that are physically close together in this space. Finding recommendations becomes finding the nearest points to a given song.

**Cosine similarity, specifically:** measures the *angle* between two vectors, not the raw distance between them. Two songs with the same relative balance of features (e.g. both "high energy, low acousticness") are considered similar even if one has slightly larger raw numbers across the board. This matters because audio features have different natural scales (tempo is in BPM, roughly 50-200; danceability is 0-1) - angle-based similarity is more robust to that than raw distance, without needing much manual rescaling.

**Why not brute-force forever:** comparing a query song against all 114,000 others by calculating similarity one-by-one works, but scales linearly - twice the songs, twice the time, every single query. Once "how similar is this to everything else" needs to happen fast and repeatedly, an index structure that's been prebuilt to answer "nearest neighbors" quickly (FAISS) replaces the brute-force loop. We'll build brute-force first (so the baseline and the "why" are both clear), then swap in FAISS and measure the actual speed difference - same "naive first, better second" teaching pattern as GateKeeper's rate limiter.

## 4. Collaborative Filtering

**Analogy:** Imagine a giant spreadsheet: rows are users, columns are artists, cells are how much each user listened to each artist. Most cells are empty (a given user's listened to a tiny fraction of all artists). Matrix factorization tries to find two smaller sets of numbers - a "taste profile" vector for each user, and a "style profile" vector for each artist - such that multiplying a user's vector by an artist's vector approximately predicts how much that user would listen to that artist, including for artists they've never listened to yet. This is how the empty cells get filled in with predictions.

**Why this works despite never being told what the songs sound like:** collaborative filtering doesn't know or care about tempo or energy. It only knows patterns in *who listened to what*. If enough users who like Artist A also like Artist B, the model learns that association purely from co-occurrence, even if A and B sound nothing alike. This is precisely what content-based filtering *can't* do (content-based only knows about the audio itself), which is why combining them is worth more than either alone.

**Cold start problem:** a brand new artist with zero listening history has no signal for collaborative filtering to learn from - the model has nothing to base a prediction on. Content-based filtering doesn't have this problem, since it only needs the song's own audio features, not any history. This is the concrete, textbook reason hybrid systems exist.

## 5. Hybrid Combination

Final recommendation score = weighted blend of the content-based similarity score and the collaborative filtering predicted score, normalized to comparable ranges first (since they come from different mathematical processes with different natural scales). Exact weighting and normalization approach finalized during implementation, with the reasoning documented once real numbers are in front of us.

## 6. Evaluation

Collaborative filtering will be evaluated with **precision@K** (of the top K recommendations, what fraction were actually in the user's held-out real listening history) - a standard recsys metric, not just "the results look plausible." This requires a train/test split on the interaction data (hide some real interactions, see if the model predicts them back).

## 7. Tech Stack & Justification

| Component | Choice | Why |
|---|---|---|
| Data handling | pandas | Standard for tabular data manipulation at this scale |
| Content-based similarity | scikit-learn (cosine_similarity), then FAISS | Baseline first, then the real scalable approach |
| Collaborative filtering | scikit-learn / implicit (ALS) | Matrix factorization designed specifically for implicit feedback (listening counts, not explicit ratings) - matches our actual data |
| API | FastAPI | Consistency with GateKeeper, and genuinely a good fit for a lightweight recommendation service |

## 8. Known Limitations (documented honestly, not hidden)

- Collaborative filtering operates at the artist level, not track level, due to real data availability (see Section 1). This is disclosed, not glossed over.
- The two datasets are not from the same underlying user base (audio features dataset has no user data at all; Last.fm dataset has no audio features) - they're joined by artist name matching, which will have some inevitable mismatches (spelling, alternate artist name formats). This is realistic - real-world recsys work constantly deals with imperfect entity resolution across data sources, and being upfront about it is more credible than pretending the join is perfect.

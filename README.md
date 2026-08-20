# Resonance

A hybrid music recommendation engine — content-based (audio feature similarity) + collaborative filtering (user listening pattern similarity).

**Status:** In progress

## Why this project

Full docs live in [`/docs`](./docs):
- [PRD](./docs/PRD.md) — what and why
- [TRD](./docs/TRD.md) — architecture, algorithms, and an honest account of data sourcing decisions (worth reading Section 1 - Spotify's live API isn't accessible for new apps, so this uses real public datasets instead)

## Data

- `data/spotify_tracks.csv` — 114,000 real Spotify tracks with audio features (content-based)
- `data/user_artists.dat` + `data/artists.dat` — HetRec 2011 Last.fm dataset, ~92,834 real user-artist listening interactions (collaborative filtering)

## Roadmap

- [ ] Content-based recommender (cosine similarity baseline)
- [ ] Collaborative filtering (matrix factorization)
- [ ] Hybrid combination
- [ ] Vector search (FAISS) for scale
- [ ] FastAPI service

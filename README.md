# Movie Recommendation System: SVD & K-Means

**Date:** November 2025

---

## Project Overview
This project implements a robust recommendation engine on **The Movies Dataset** (26 million ratings). It tackles the challenge of extreme data sparsity (99.79%) by comparing two major approaches:
1.  **Collaborative Filtering (via SVD):** Predicting missing ratings by factorizing the user-item interaction matrix.
2.  **Unsupervised Clustering (via K-Means):** Attempting to group users by taste profiles.

## Methodology

### 1. Model-based Collaborative Filtering (SVD)
Instead of traditional memory-based methods (like Nearest Neighbors), we utilized **Singular Value Decomposition (SVD)**, a matrix factorization technique.
* **Concept:** Decomposed the sparse rating matrix $R$ into latent feature matrices ($U$ and $V^T$).
* **Implementation:** Used `scipy.sparse.linalg.svds` with **k=70** latent factors to capture underlying user-item interactions.
* **Why CF?** By learning latent factors from collective user behaviors, the model successfully predicts ratings for unseen movies.

### 2. User Segmentation (K-Means)
* Applied **MiniBatch K-Means** to segment users based on their latent feature vectors.
* **Insight:** The "Elbow Method" suggested $K=3$, but analysis showed that user preferences are distributed on a continuous manifold rather than in distinct "market segments."

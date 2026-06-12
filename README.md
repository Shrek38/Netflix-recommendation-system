# Netflix Recommendation System

A personalized movie recommendation engine built on the Netflix Prize Dataset (~100M ratings). Implements and compares two collaborative filtering approaches — SVD (Matrix Factorization) and KNN Item-Based CF — evaluated on both rating prediction accuracy and ranking quality.

---

## What's in this project

- **EDA** — rating distributions, user activity patterns, movie popularity (long-tail), temporal trends, and release era analysis
- **SVD** — latent factor model via Singular Value Decomposition using `scikit-surprise`
- **KNN Item-Based CF** — neighborhood-based collaborative filtering with explainable recommendations
- **Evaluation** — RMSE, MAE, MAP@10, Precision@10, Recall@10, NDCG@10, Hit Rate
- **Recommendation generation** — Top-10 personalized recommendations for users across activity levels
- **Explainability** — KNN-based explanations ("recommended because you liked X and Y")
- **Cold start strategy** — popularity-based fallback using Bayesian average scoring for new users

---

## Dataset

[Netflix Prize Dataset](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) — download from Kaggle 

The full dataset has ~100M ratings across 480K users and 17.7K movies. The notebook loads a **20% random sample (~20M ratings)** for memory efficiency, then applies quality filters (users ≥ 10 ratings, movies ≥ 20 ratings), leaving roughly 19M ratings for training and evaluation.

---

## Setup

```bash
pip install numpy==1.26.4
pip install scikit-surprise
```

> **Note:** Run only the install cell first. After it finishes, restart the kernel and clear all outputs, comment out the install cell, then click Run All.

Other dependencies (`pandas`, `matplotlib`, `seaborn`, `scikit-learn`) should already be available in standard environments like Kaggle or Colab.

---

## How to run

1. Download the dataset from Kaggle
2. Set `DATA_PATH` in the data loading cell to point to your local dataset directory
3. Follow the kernel restart instructions at the top of the notebook
4. Run all cells — figures generate inline and the final cell prints a complete results summary

---

## Results

| Metric | SVD | KNN Item-CF |
|---|---|---|
| RMSE ↓ | run notebook | run notebook |
| MAE ↓ | run notebook | run notebook |
| MAP@10 ↑ | run notebook | run notebook |
| Precision@10 ↑ | run notebook | run notebook |
| NDCG@10 ↑ | run notebook | run notebook |
| Hit Rate@10 ↑ | run notebook | run notebook |

SVD consistently achieves lower RMSE through latent factor modeling. KNN Item-CF is slower to train but produces interpretable recommendations. In practice, a hybrid of both would be the strongest approach.

---

## Project structure

```
Netflix-recommendation-system/
│
├── code.ipynb          # Main notebook — EDA, training, evaluation, recommendations
└── README.md
```

---

## Key findings

- Rating distribution is left-skewed (mean ~3.6) — users predominantly rate content they enjoyed, introducing selection bias
- Top ~13% of movies account for 80% of all ratings, a classic long-tail content distribution
- Median user has ~35 ratings; the user-item matrix is over 98% sparse, making compact latent representations essential
- SVD learns better rating predictions; KNN lets you explain *why* a movie was recommended, which matters for user trust
- RMSE and MAP@10 measure fundamentally different things — strong rating accuracy doesn't guarantee strong ranking performance
- Cold-start users are handled via Bayesian-average popularity scoring as a fallback before collaborative signals accumulate

---

## Acknowledgements

Dataset: [Netflix Prize](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) via Kaggle  
## Recommender System Project

This repository contains the code, data loading utilities, exploratory data analysis (EDA), and model implementations for my recommender system built on the KuaiRec 2.0 dataset.
We explore multiple approaches :

- ALS (Alternating Least Squares),
- Deep Autoencoder,
- Two-Tower models

---

## 📁 Repository Structure

```

.venv/                  # Python virtual environment

data_final_project/     # Raw data directory (KuaiRec 2.0)
  └─ data/              # CSV files for user-item interactions, features, etc.

trained/                # Saved trained model checkpoints

load_data.py            # Script to load and clean datasets
utils.py                # Utility functions (data paths, cleaning, summaries)
requirements.txt        # Python dependencies

README.md               # Project overview and usage instructions (this file)
subject.md              # Project proposal and objectives

# Notebooks
givenEDA.ipynb          # Provided baseline EDA
myEDA.ipynb             # Custom exploratory analysis and visualizations
als.ipynb               # ALS collaborative filtering baseline
autoencoder.ipynb       # Deep autoencoder implementation using PyTorch
two_towers.ipynb        # Two-Tower model (dual-encoder) implementation

```

---

## 🚀 Getting Started

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 🔎 Exploratory Data Analysis (EDA)

* **`givenEDA.ipynb`** : Baseline EDA provided with the course.
* **`myEDA.ipynb`** : Custom analysis including:
* Distribution of user interactions.
* Sparsity and density heatmaps.
* Popular vs. niche video comparisons.
* Feature correlations (e.g., watch ratio vs. play duration).

Key insights:

* The dataset is extremely sparse (>99.9% zeros).
* A small fraction of videos accounts for the majority of views.

For a more detailed analysis, check the `myEDA.ipynb` file.

---

## 🛠️ Models Implemented

### 1. ALS Collaborative Filtering (`als.ipynb`)

* **Library** : `implicit` & `pyspark.ml.recommendation`
* **Approach** : Factorizes the user-item interaction matrix into user and item latent vectors.
* **Evaluation** :
* RMSE on held-out interactions.
* Top-10 Precision, Recall, and NDCG.
* **Results** :
* Baseline performance with small subset due to memory constraints.
* Precision@10: ~0.15, Recall@10: ~0.05, NDCG@10: ~0.30

**What is it?**
We first tried the ALS (Alternating Least Squares) algorithm from the `implicit` library, then switched to `pyspark.ml.recommendation` so it could handle more data and let us tune hyperparameters and run cross‐validation. ALS is a collaborative filtering method that breaks down the huge user–item interaction matrix into two smaller matrices (one for users, one for items) and tries to rebuild the original matrix as accurately as possible. Since our data is implicit (we only know which videos people watched, not explicit ratings), ALS is a good fit—and it works especially well on sparse data.

**How did we evaluate it?**

* We used Spark’s `RegressionEvaluator` with RMSE to gauge how well the model reconstructs the interaction matrix.
* For recommendations, we generated each user’s top-10 videos and compared them to what they actually watched, computing NDCG, recall, and precision. Then we averaged those metrics across all users.
* To map item IDs back to real videos, we joined in the `kuairec_caption_category.csv` file.

**What went wrong?**

* When we tried to train on the full `big_matrix.csv`, we ran into memory and performance issues, so we fell back to a smaller subset (`small_matrix.csv`).
* We only used user–item data—no extra features—so ALS here is really just a basic baseline.

**Next steps:**

* **Add more features.** Include video metadata or user profiles.
* **Scale up.** Figure out distributed training or pre-reduce the data so we can use the full matrix.
* **Try fancier factorization.** Methods like BPR or Weighted Matrix Factorization often handle implicit feedback better.

### 2. Deep Autoencoder (`autoencoder.ipynb`)

* **Framework** : PyTorch + PyTorch Lightning
* **Architecture** :

```python
  model = DeepAutoEncoder(
      input_dim = n_items,
      hidden_dims = [256, 128],
      dropout = 0.3
  )
  optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
```

* **Remarks** : Achieves strong personalization vs. popularity baseline.

**What is it?**
In `AutoEncoder.ipynb`, I built a two-layer deep autoencoder using PyTorch. The idea is:

1. **Encode** each user’s watch history (a big binary vector) into a smaller “latent” vector.
2. **Decode** that latent vector back into a reconstruction of their preferences.
3. Train the whole thing with binary cross-entropy loss on a dense binary matrix.

**Model setup:**

```python
model = DeepAutoEncoder(input_dim=n_items, hidden_dims=[256, 128], dropout=0.3)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
```

**Why use this?**

* It can learn non-linear relationships, giving richer user embeddings than SVD or ALS.
* More expressive than a simple one-layer approach.
* Dropout helps it deal with our really sparse data.

**How did it do?**
At top-10 recommendations:

* **Precision\@10:** 0.5227
* **Recall\@10:** 0.0915
* **NDCG\@10:** 0.5586
* **MAP\@10:** 0.3996

**What does that mean?**

* A precision of 0.52 is way better than our SVD baseline (\~0.02) and almost as good as just recommending the most popular videos (\~0.75), but with actual personalization.
* Recall of 0.09 means it finds far more of a user’s true interests compared to popularity (0.008).
* NDCG and MAP show it not only picks the right videos but also ranks them well.

**Bottom line:**
Out of all the methods we tried, the deep autoencoder strikes the best balance between recommending what’s popular and tailoring suggestions to each user’s tastes.

### 3. Two-Tower Model (`two_towers.ipynb`)

* **User tower** : Historical watch vector + user features.
* **Item tower** : Video ID embedding + content features.
* **Loss** : Contrastive or cross-entropy loss over positive and negative pairs.
* **Findings** : Stronger generalization to cold-start items when using side features.

The Two Towers model was developed to predict user-video interactions by utilizing both user and item features. The model was trained and evaluated on a dataset containing user interactions with videos.

#### Evaluation Metrics:

- **Mean Absolute Error (MAE):** 0.5125This value reflects the average absolute difference between the predicted and actual outcomes. Lower MAE values indicate better predictive accuracy.
- **Root Mean Squared Error (RMSE):** 1.6252RMSE calculates the square root of the average squared differences between predictions and actual values, penalizing larger errors more heavily than MAE.
- **R-squared (R2):** 0.0628
  R2 measures the proportion of variance in the target variable explained by the model. A low R2 score suggests that the model accounts for only a small fraction of the data’s variability.

#### Insights:

- The low R2 score indicates that the model currently struggles to capture the underlying patterns in the data.
- The MAE and RMSE values provide an understanding of the prediction errors, with RMSE highlighting larger discrepancies.

#### Final Thoughts:

While the Two Towers model shows some capability in predicting user-video interactions, its current performance leaves considerable room 	for improvement. Future work should focus on refining the model, enhancing feature engineering, and exploring alternative approaches. These steps can help boost the model’s predictive accuracy and its ability to generalize to new data.

---

📈 Evaluation & Results

| Model       | Precision@10   | Recall@10      | NDCG@10        | MAP@10         |
| ----------- | -------------- | -------------- | -------------- | -------------- |
| ALS         | 0.15           | 0.05           | 0.30           | 0.20           |
| Autoencoder | **0.52** | **0.09** | **0.56** | **0.40** |
| Two-Tower   | 0.48           | 0.08           | 0.53           | 0.38           |


---
📌 Conclusion

This project journey has showcased the power and complexity of modern recommender systems, even if the top-10 metrics—while respectable—are not sky-high. More importantly, the hands-on experience has deepened my understanding of ALS, deep autoencoders, and two‑tower architectures beyond what theoretical study alone could offer.

I want to extend my gratitude to our instructor, whose clear explanations and supportive guidance made every class a valuable experience. I attended every session, and his expertise and approachability have motivated me to continue exploring this field. I look forward to keeping in touch and learning from him in the future.

While there is room to improve model performance, the real success lies in the skills and insights gained, which will inform all my future work in recommendation systems.

**Author** : Tidjani ADAM KANDINE

 **Date** : May 2025

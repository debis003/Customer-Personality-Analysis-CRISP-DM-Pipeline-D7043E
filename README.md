# Customer Personality Analysis — CRISP-DM Pipeline (D7043E)

Group project from **Advanced Data Mining (D7043E)** at Luleå University of Technology (2025).

We followed the full **CRISP-DM methodology** on a customer dataset (2,216 records) to tackle two goals:

- **Classification** — Predicted campaign response using Logistic Regression and Random Forest, achieving 0.91 AUC and 89% accuracy.
- **Clustering** — Segmented customers into 5 personas using Deep Embedded Clustering (DEC), reaching 0.88 silhouette score — far outperforming the Autoencoder + K-Means baseline.

Key work included engineering 51 behavioral/demographic features from the original 27, handling class imbalance, and translating cluster profiles into actionable marketing recommendations.

The source code was developed collaboratively in [bardia2532/Advanced-Data-Mining-D7043E](https://github.com/bardia2532/Advanced-Data-Mining-D7043E). This repository contains the final presentation and report.

**Technologies**: Python, scikit-learn, PyTorch, Pandas, Matplotlib, RFM Analysis, Deep Embedded Clustering

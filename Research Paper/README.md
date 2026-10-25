# ══════════════════════════════════════════════════════════════
#  Mini Project — DWDM (Semester 5)
#  Customer Segmentation and Fraud Detection in Banking
#  Using Data Mining Techniques
# ══════════════════════════════════════════════════════════════

## Project Overview

This project applies **data mining techniques** to two critical 
banking problems:

1. **Customer Segmentation** — Identifying distinct customer groups 
   using unsupervised clustering algorithms
2. **Fraud Detection** — Detecting fraudulent transactions using 
   supervised and anomaly detection algorithms

## Algorithms Implemented

### Part A: Customer Segmentation (Unsupervised Learning)
| # | Algorithm | Type |
|---|-----------|------|
| 1 | K-Means Clustering | Centroid-based |
| 2 | Hierarchical (Agglomerative) Clustering | Linkage-based |
| 3 | DBSCAN | Density-based |
| 4 | Gaussian Mixture Model (GMM) | Distribution-based |

### Part B: Fraud Detection (Supervised + Anomaly Detection)
| # | Algorithm | Type |
|---|-----------|------|
| 5 | Logistic Regression | Linear classifier |
| 6 | Random Forest | Ensemble (Bagging) |
| 7 | Isolation Forest | Anomaly detection |
| 8 | XGBoost | Ensemble (Boosting) |

## Datasets Used
1. **CC GENERAL** (Kaggle) — 8,950 credit card customers, 18 features
2. **Credit Card Fraud** (Kaggle/ULB) — 284,807 transactions
3. **Default of Credit Card Clients** (UCI) — 30,000 observations

See `DATASET_DESCRIPTION.md` for full details.

## Project Structure

```
Research paper/
├── README.md                    # This file
├── DATASET_DESCRIPTION.md       # Detailed dataset descriptions
├── code/
│   └── main.py          # Complete project code (all algorithms)
├── datasets/
│   ├── dataset1_cc_general.csv          # Downloaded/generated on run
│   ├── dataset2_fraud_detection.csv     # Downloaded/generated on run
│   ├── dataset3_default_clients.csv     # Downloaded/generated on run
│   ├── preprocessed_segmentation.csv    # After preprocessing
│   └── preprocessed_fraud.csv           # After preprocessing
├── outputs/
│   ├── segmentation/
│   │   ├── 01_eda_cc_general.png
│   │   ├── 02_correlation_heatmap.png
│   │   ├── 03_kmeans_analysis.png
│   │   ├── 04_hierarchical_analysis.png
│   │   ├── 05_dbscan_analysis.png
│   │   ├── 06_gmm_analysis.png
│   │   └── 07_segmentation_comparison.png
│   ├── fraud_detection/
│   │   ├── logistic_regression_eval.png
│   │   ├── random_forest_eval.png
│   │   ├── isolation_forest_eval.png
│   │   ├── xgboost_eval.png
│   │   ├── rf_feature_importance.png
│   │   └── fraud_comparison_all.png
│   └── comparison/
│       ├── segmentation_comparison.csv
│       ├── fraud_comparison.csv
│       └── overall_project_summary.png
└── paper/
    └── (Research paper files - .docx and .pdf)
```

## How to Run

### Prerequisites
```
Python 3.8+
pip install numpy pandas matplotlib seaborn scikit-learn scipy xgboost
```

### Run the complete project
```bash
cd Research paper/code
python main.py
```

The script will:
1. Download datasets from online sources (with fallback to synthetic data)
2. Perform EDA and preprocessing
3. Run all 8 algorithms
4. Generate all evaluation plots
5. Save comparison tables and summary charts

## Evaluation Metrics

### Segmentation Metrics
- **Silhouette Score** (Higher = Better) — Measures cluster cohesion and separation
- **Davies-Bouldin Index** (Lower = Better) — Average similarity between clusters
- **Calinski-Harabasz Score** (Higher = Better) — Ratio of between-cluster to within-cluster dispersion

### Fraud Detection Metrics
- **Accuracy** — Overall correctness
- **Precision** — True positives / (True positives + False positives)
- **Recall** — True positives / (True positives + False negatives)
- **F1-Score** — Harmonic mean of Precision and Recall
- **ROC-AUC** — Area Under the ROC Curve
- **Average Precision** — Area Under the Precision-Recall Curve

## Team Members
1. Member 1 (Team Leader)
2. Member 2
3. Member 3
4. Member 4
5. Member 5

## References
See the research paper for complete references (30+ citations).

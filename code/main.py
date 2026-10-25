# ================================================================================
#  MINI PROJECT: Customer Segmentation and Clustering in Banking
#                Using Data Mining Techniques
#
#  Course  : Data Warehousing and Data Mining (DWDM) - Semester 5
#  Datasets: 1. Credit Card Customer Segmentation (Kaggle - CC GENERAL)
#             2. Credit Card Fraud Detection (Kaggle - ULB)
#             3. Default of Credit Card Clients (UCI ML Repository)
#
#  Part A - Customer Segmentation (Unsupervised Learning)
#     Algorithms: K-Means, Hierarchical (Agglomerative), DBSCAN, GMM
#
#  Part B - Fraud Detection (Supervised + Anomaly Detection)
#     Algorithms: Logistic Regression, Random Forest, Isolation Forest, XGBoost
# ================================================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')                 # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    f1_score, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# Try importing XGBoost (install if missing)
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    print("[WARN]  XGBoost not installed. Installing...")
    os.system(f"{sys.executable} -m pip install xgboost -q")
    from xgboost import XGBClassifier
    HAS_XGBOOST = True

# -- Output directories ------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEG_OUT     = os.path.join(BASE_DIR, "outputs", "segmentation")
FRAUD_OUT   = os.path.join(BASE_DIR, "outputs", "fraud_detection")
COMP_OUT    = os.path.join(BASE_DIR, "outputs", "comparison")
DATA_DIR    = os.path.join(BASE_DIR, "datasets")
os.makedirs(SEG_OUT,   exist_ok=True)
os.makedirs(FRAUD_OUT, exist_ok=True)
os.makedirs(COMP_OUT,  exist_ok=True)
os.makedirs(DATA_DIR,  exist_ok=True)

# -- Plot styling ------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
COLORS = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12',
          '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b']

print("=" * 80)
print("  MINI PROJECT: Customer Segmentation & Fraud Detection in Banking")
print("  Using Data Mining Techniques")
print("=" * 80)

# +===========================================================================+
# |                    SECTION 1 : DATA LOADING                              |
# +===========================================================================+

print("\n" + "-" * 80)
print("  SECTION 1 : LOADING DATASETS FROM LOCAL FILES")
print("-" * 80)

# --- Dataset 1: Credit Card Customer Data (CC GENERAL - Kaggle) ----------
# Source: https://www.kaggle.com/datasets/arjunbhasin2013/ccdata
# ~8950 active credit card holders, 18 behavioral features
print("\n[Dataset 1] Credit Card Customer Behavioral Data (CC GENERAL)")
df_cc = pd.read_csv(os.path.join(DATA_DIR, "CC_GENERAL.csv"))
print(f"   Loaded from local file  ->  {df_cc.shape}")

# --- Dataset 2: Credit Card Fraud Detection (Kaggle - ULB) --------------
# Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# 284,807 transactions, PCA features V1-V28 + Time + Amount + Class
print("\n[Dataset 2] Credit Card Fraud Detection (European Transactions)")
df_fraud_raw = pd.read_csv(os.path.join(DATA_DIR, "creditcard.csv"), nrows=20000)
print(f"   Loaded from local file  ->  {df_fraud_raw.shape}")

# --- Dataset 3: Default of Credit Card Clients (UCI ML Repository) ------
# Source: https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
# 30,000 observations with demographics + payment history
print("\n[Dataset 3] Default of Credit Card Clients (Taiwan)")
df_default = pd.read_csv(os.path.join(DATA_DIR, "UCI_Credit_Card.csv"), nrows=15000)
print(f"   Loaded from local file  ->  {df_default.shape}")

# --- Save raw datasets -----------------------------------------------------
df_cc.to_csv(os.path.join(DATA_DIR, "dataset1_cc_general.csv"), index=False)
df_fraud_raw.to_csv(os.path.join(DATA_DIR, "dataset2_fraud_detection.csv"), index=False)
df_default.to_csv(os.path.join(DATA_DIR, "dataset3_default_clients.csv"), index=False)
print(f"\n[SAVE] All 3 datasets saved to: {DATA_DIR}")


# +===========================================================================+
# |                 SECTION 2 : EXPLORATORY DATA ANALYSIS                    |
# +===========================================================================+

print("\n" + "-" * 80)
print("  SECTION 2 : EXPLORATORY DATA ANALYSIS (EDA)")
print("-" * 80)

# --- EDA: Dataset 1 (CC GENERAL) ----------------------------------------
print("\n-- Dataset 1: CC GENERAL --")
print(f"   Shape        : {df_cc.shape}")
print(f"   Missing vals : {df_cc.isnull().sum().sum()}")
print(f"   Columns      : {list(df_cc.columns)}")
print(df_cc.describe().round(2))

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Dataset 1 - Credit Card Customer Behavioral Data (EDA)", fontsize=16, fontweight='bold')

# Distribution of key features
num_cols_cc = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS', 'TENURE']
for i, col in enumerate(num_cols_cc):
    ax = axes[i//3][i%3]
    if col in df_cc.columns:
        df_cc[col].hist(bins=40, ax=ax, color=COLORS[i], alpha=0.75, edgecolor='white')
        ax.set_title(col, fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(SEG_OUT, "01_eda_cc_general.png"), dpi=150, bbox_inches='tight')
plt.close()
print("   [CHART] 01_eda_cc_general.png saved")

# Correlation heatmap
fig, ax = plt.subplots(figsize=(14, 10))
numeric_cc = df_cc.select_dtypes(include=[np.number])
corr = numeric_cc.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r', ax=ax,
            linewidths=0.5, square=True, cbar_kws={'shrink': 0.8},
            annot_kws={'size': 7})
ax.set_title("Dataset 1 - Correlation Heatmap", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SEG_OUT, "02_correlation_heatmap.png"), dpi=150, bbox_inches='tight')
plt.close()
print("   [CHART] 02_correlation_heatmap.png saved")


# +===========================================================================+
# |               SECTION 3 : DATA PREPROCESSING                             |
# +===========================================================================+

print("\n" + "-" * 80)
print("  SECTION 3 : DATA PREPROCESSING")
print("-" * 80)

# --- Preprocess Dataset 1 for Segmentation ------------------------------
print("\n-- Preprocessing Dataset 1 (CC GENERAL) for Segmentation --")

# Drop non-numeric ID column
if 'CUST_ID' in df_cc.columns:
    df_seg = df_cc.drop('CUST_ID', axis=1)
else:
    df_seg = df_cc.copy()

# Handle missing values - fill with median
missing_before = df_seg.isnull().sum().sum()
df_seg = df_seg.fillna(df_seg.median(numeric_only=True))
missing_after = df_seg.isnull().sum().sum()
print(f"   Missing values: {missing_before} -> {missing_after}")

# Remove outliers using IQR method (cap at 1.5 x IQR)
for col in df_seg.select_dtypes(include=[np.number]).columns:
    Q1, Q3 = df_seg[col].quantile(0.25), df_seg[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df_seg[col] = df_seg[col].clip(lower, upper)

# Feature scaling - StandardScaler
scaler_seg = StandardScaler()
X_seg = scaler_seg.fit_transform(df_seg.select_dtypes(include=[np.number]))
print(f"   Feature matrix shape: {X_seg.shape}")
print(f"   Features used       : {list(df_seg.select_dtypes(include=[np.number]).columns)}")

# PCA for visualization (reduce to 2D)
pca_seg = PCA(n_components=2)
X_seg_2D = pca_seg.fit_transform(X_seg)
print(f"   PCA Variance Explained: {pca_seg.explained_variance_ratio_.sum():.4f} "
      f"({pca_seg.explained_variance_ratio_[0]:.3f} + {pca_seg.explained_variance_ratio_[1]:.3f})")


# --- Preprocess Dataset 2 + 3 for Fraud Detection ----------------------
print("\n-- Preprocessing Datasets 2 & 3 for Fraud Detection --")

# Dataset 2: Fraud Detection
df_fraud = df_fraud_raw.copy()
# Normalize Amount and Time
df_fraud['Amount_Scaled'] = StandardScaler().fit_transform(df_fraud[['Amount']])
df_fraud['Time_Scaled']   = StandardScaler().fit_transform(df_fraud[['Time']])

# Dataset 3: Default of Credit Card - extract relevant features as
# supplementary "default risk" indicators
# We will create an enriched fraud dataset by engineering features from
# the default dataset that are aligned with the fraud detection task

# Build fraud detection feature matrix
fraud_features = [c for c in df_fraud.columns if c.startswith('V')] + ['Amount_Scaled', 'Time_Scaled']
X_fraud = df_fraud[fraud_features].values
y_fraud = df_fraud['Class'].astype(int).values

# Handle any remaining NaN
X_fraud = np.nan_to_num(X_fraud, nan=0.0)

# Train / test split (stratified due to class imbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X_fraud, y_fraud, test_size=0.20, stratify=y_fraud, random_state=42
)

# Scale
scaler_fraud = StandardScaler()
X_train_sc = scaler_fraud.fit_transform(X_train)
X_test_sc  = scaler_fraud.transform(X_test)

n_legit = int((y_fraud == 0).sum())
n_fraud_c = int((y_fraud == 1).sum())
print(f"   Total samples     : {len(y_fraud)}")
print(f"   Legitimate        : {n_legit} ({n_legit/len(y_fraud)*100:.1f}%)")
print(f"   Fraudulent        : {n_fraud_c} ({n_fraud_c/len(y_fraud)*100:.1f}%)")
print(f"   Train shape       : {X_train_sc.shape}")
print(f"   Test shape        : {X_test_sc.shape}")

# Save preprocessed data
pd.DataFrame(X_seg, columns=df_seg.select_dtypes(include=[np.number]).columns).to_csv(
    os.path.join(DATA_DIR, "preprocessed_segmentation.csv"), index=False)
pd.DataFrame(X_fraud, columns=fraud_features).to_csv(
    os.path.join(DATA_DIR, "preprocessed_fraud.csv"), index=False)
print(f"\n   [SAVE] Preprocessed data saved")


# +===========================================================================+
# |         PART A : CUSTOMER SEGMENTATION (Unsupervised Learning)           |
# +===========================================================================+

print("\n" + "=" * 80)
print("  PART A : CUSTOMER SEGMENTATION (Unsupervised Learning)")
print("=" * 80)

seg_results = {}

# ----------------------------------------------------------------------
#  ALGORITHM 1 : K-MEANS CLUSTERING
# ----------------------------------------------------------------------
print("\n" + "-" * 60)
print("  Algorithm 1 : K-MEANS CLUSTERING")
print("-" * 60)

inertias, sil_km, dbi_km, chi_km = [], [], [], []
K_range = range(2, 11)

for k in K_range:
    km_ = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    km_.fit(X_seg)
    inertias.append(km_.inertia_)
    sil_km.append(silhouette_score(X_seg, km_.labels_))
    dbi_km.append(davies_bouldin_score(X_seg, km_.labels_))
    chi_km.append(calinski_harabasz_score(X_seg, km_.labels_))

opt_k = K_range[int(np.argmax(sil_km))]
km = KMeans(n_clusters=opt_k, random_state=42, n_init=10, max_iter=300)
km_labels = km.fit_predict(X_seg)

sil_km_best = silhouette_score(X_seg, km_labels)
dbi_km_best = davies_bouldin_score(X_seg, km_labels)
chi_km_best = calinski_harabasz_score(X_seg, km_labels)

seg_results['K-Means'] = {
    'clusters': opt_k, 'silhouette': sil_km_best,
    'davies_bouldin': dbi_km_best, 'calinski_harabasz': chi_km_best,
    'labels': km_labels
}

print(f"   Optimal k           : {opt_k}")
print(f"   Silhouette Score    : {sil_km_best:.4f}")
print(f"   Davies-Bouldin Index: {dbi_km_best:.4f}")
print(f"   Calinski-Harabasz   : {chi_km_best:.2f}")
print(f"   Inertia             : {km.inertia_:.2f}")

# K-Means Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle("K-Means Clustering Analysis", fontsize=16, fontweight='bold')

# Elbow Method
axes[0][0].plot(K_range, inertias, 'o-', color='#3498db', lw=2.5, markersize=8)
axes[0][0].axvline(opt_k, color='#e74c3c', ls='--', lw=2, label=f'Optimal k={opt_k}')
axes[0][0].set_title('Elbow Method (Inertia vs K)', fontsize=13, fontweight='bold')
axes[0][0].set_xlabel('Number of Clusters (k)', fontsize=11)
axes[0][0].set_ylabel('Inertia (WCSS)', fontsize=11)
axes[0][0].legend(fontsize=11); axes[0][0].grid(True, alpha=0.3)

# Silhouette Scores
axes[0][1].plot(K_range, sil_km, 's-', color='#2ecc71', lw=2.5, markersize=8)
axes[0][1].axvline(opt_k, color='#e74c3c', ls='--', lw=2, label=f'Best k={opt_k}')
axes[0][1].set_title('Silhouette Score vs K', fontsize=13, fontweight='bold')
axes[0][1].set_xlabel('Number of Clusters (k)', fontsize=11)
axes[0][1].set_ylabel('Silhouette Score', fontsize=11)
axes[0][1].legend(fontsize=11); axes[0][1].grid(True, alpha=0.3)

# PCA 2D Scatter
scatter_km = axes[1][0].scatter(X_seg_2D[:, 0], X_seg_2D[:, 1],
                                 c=km_labels, cmap='tab10', s=15, alpha=0.6)
centers_2D = pca_seg.transform(km.cluster_centers_)
axes[1][0].scatter(centers_2D[:, 0], centers_2D[:, 1],
                    c='black', marker='X', s=250, edgecolors='white', linewidths=2,
                    label='Centroids', zorder=5)
axes[1][0].set_title(f'K-Means Clusters (k={opt_k}) - PCA Projection', fontsize=13, fontweight='bold')
axes[1][0].set_xlabel(f'PC1 ({pca_seg.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
axes[1][0].set_ylabel(f'PC2 ({pca_seg.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
axes[1][0].legend(fontsize=11); axes[1][0].grid(True, alpha=0.3)
plt.colorbar(scatter_km, ax=axes[1][0], label='Cluster')

# Cluster Distribution
cluster_counts = pd.Series(km_labels).value_counts().sort_index()
bars = axes[1][1].bar(cluster_counts.index, cluster_counts.values,
                       color=[COLORS[i % len(COLORS)] for i in range(len(cluster_counts))],
                       edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, cluster_counts.values):
    axes[1][1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                     str(val), ha='center', fontweight='bold', fontsize=11)
axes[1][1].set_title('Cluster Size Distribution', fontsize=13, fontweight='bold')
axes[1][1].set_xlabel('Cluster', fontsize=11); axes[1][1].set_ylabel('Count', fontsize=11)
axes[1][1].grid(True, alpha=0.3, axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(SEG_OUT, "03_kmeans_analysis.png"), dpi=150, bbox_inches='tight')
plt.close()
print("   [CHART] 03_kmeans_analysis.png saved")


# ----------------------------------------------------------------------
#  ALGORITHM 2 : HIERARCHICAL (AGGLOMERATIVE) CLUSTERING
# ----------------------------------------------------------------------
print("\n" + "-" * 60)
print("  Algorithm 2 : HIERARCHICAL (AGGLOMERATIVE) CLUSTERING")
print("-" * 60)

# Use same k as K-Means for fair comparison
hc = AgglomerativeClustering(n_clusters=opt_k, linkage='ward')
hc_labels = hc.fit_predict(X_seg)

sil_hc = silhouette_score(X_seg, hc_labels)
dbi_hc = davies_bouldin_score(X_seg, hc_labels)
chi_hc = calinski_harabasz_score(X_seg, hc_labels)

seg_results['Hierarchical'] = {
    'clusters': opt_k, 'silhouette': sil_hc,
    'davies_bouldin': dbi_hc, 'calinski_harabasz': chi_hc,
    'labels': hc_labels
}

print(f"   Clusters            : {opt_k} (Ward linkage)")
print(f"   Silhouette Score    : {sil_hc:.4f}")
print(f"   Davies-Bouldin Index: {dbi_hc:.4f}")
print(f"   Calinski-Harabasz   : {chi_hc:.2f}")

# Hierarchical Visualization
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Hierarchical (Agglomerative) Clustering Analysis", fontsize=16, fontweight='bold')

# Dendrogram (on a sample for readability)
sample_idx = np.random.choice(len(X_seg), min(80, len(X_seg)), replace=False)
Z = linkage(X_seg[sample_idx], method='ward')
dendrogram(Z, ax=axes[0], leaf_rotation=90, leaf_font_size=6,
           color_threshold=Z[-opt_k+1, 2], above_threshold_color='gray')
axes[0].axhline(Z[-opt_k+1, 2], color='red', ls='--', lw=2, label=f'Cut at k={opt_k}')
axes[0].set_title(f'Dendrogram (Ward Linkage, {len(sample_idx)} samples)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Sample Index'); axes[0].set_ylabel('Distance')
axes[0].legend(fontsize=11)

# PCA 2D Scatter
scatter_hc = axes[1].scatter(X_seg_2D[:, 0], X_seg_2D[:, 1],
                               c=hc_labels, cmap='tab10', s=15, alpha=0.6)
axes[1].set_title(f'Hierarchical Clusters (k={opt_k}) - PCA Projection', fontsize=13, fontweight='bold')
axes[1].set_xlabel(f'PC1 ({pca_seg.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
axes[1].set_ylabel(f'PC2 ({pca_seg.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
axes[1].grid(True, alpha=0.3)
plt.colorbar(scatter_hc, ax=axes[1], label='Cluster')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(SEG_OUT, "04_hierarchical_analysis.png"), dpi=150, bbox_inches='tight')
plt.close()
print("   [CHART] 04_hierarchical_analysis.png saved")


# ----------------------------------------------------------------------
#  ALGORITHM 3 : DBSCAN (Density-Based Spatial Clustering)
# ----------------------------------------------------------------------
print("\n" + "-" * 60)
print("  Algorithm 3 : DBSCAN")
print("-" * 60)

# Grid search for best eps and min_samples
best_sil_db, best_eps, best_ms = -1, 0.5, 5
for eps in [0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0]:
    for ms in [3, 5, 8, 10, 15]:
        lbl_ = DBSCAN(eps=eps, min_samples=ms).fit_predict(X_seg)
        nc_ = len(set(lbl_)) - (1 if -1 in lbl_ else 0)
        noise_ratio = list(lbl_).count(-1) / len(lbl_)
        if nc_ >= 2 and noise_ratio < 0.30:
            try:
                s_ = silhouette_score(X_seg, lbl_)
                if s_ > best_sil_db:
                    best_sil_db, best_eps, best_ms = s_, eps, ms
            except:
                pass

db_labels = DBSCAN(eps=best_eps, min_samples=best_ms).fit_predict(X_seg)
nc_db  = len(set(db_labels)) - (1 if -1 in db_labels else 0)
nn_db  = list(db_labels).count(-1)

if nc_db >= 2:
    mask_db = db_labels != -1
    sil_db = silhouette_score(X_seg[mask_db], db_labels[mask_db]) if mask_db.sum() > 0 else 0
    dbi_db = davies_bouldin_score(X_seg[mask_db], db_labels[mask_db]) if mask_db.sum() > 0 else 0
    chi_db = calinski_harabasz_score(X_seg[mask_db], db_labels[mask_db]) if mask_db.sum() > 0 else 0
else:
    sil_db, dbi_db, chi_db = 0, 0, 0

seg_results['DBSCAN'] = {
    'clusters': nc_db, 'silhouette': sil_db,
    'davies_bouldin': dbi_db, 'calinski_harabasz': chi_db,
    'labels': db_labels
}

print(f"   Best eps            : {best_eps}")
print(f"   Best min_samples    : {best_ms}")
print(f"   Clusters found      : {nc_db}")
print(f"   Noise points        : {nn_db} ({nn_db/len(db_labels)*100:.1f}%)")
print(f"   Silhouette Score    : {sil_db:.4f}")
print(f"   Davies-Bouldin Index: {dbi_db:.4f}")
print(f"   Calinski-Harabasz   : {chi_db:.2f}")

# DBSCAN Visualization
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("DBSCAN Clustering Analysis", fontsize=16, fontweight='bold')

# Cluster plot with noise highlighted
unique_labels = set(db_labels)
for label in sorted(unique_labels):
    mask = db_labels == label
    if label == -1:
        axes[0].scatter(X_seg_2D[mask, 0], X_seg_2D[mask, 1],
                         c='red', marker='x', s=30, alpha=0.4, label='Noise')
    else:
        axes[0].scatter(X_seg_2D[mask, 0], X_seg_2D[mask, 1],
                         c=COLORS[label % len(COLORS)], s=15, alpha=0.6,
                         label=f'Cluster {label}')
axes[0].set_title(f'DBSCAN Clusters (eps={best_eps}, min_samples={best_ms})',
                   fontsize=13, fontweight='bold')
axes[0].set_xlabel(f'PC1', fontsize=11); axes[0].set_ylabel(f'PC2', fontsize=11)
axes[0].legend(fontsize=9, loc='best'); axes[0].grid(True, alpha=0.3)

# Cluster size distribution
if nc_db > 0:
    cl_sizes = pd.Series(db_labels[db_labels != -1]).value_counts().sort_index()
    noise_count = pd.Series({'Noise': nn_db})
    all_sizes = pd.concat([cl_sizes.rename({i: f'C{i}' for i in cl_sizes.index}), noise_count])
    colors_bar = [COLORS[i % len(COLORS)] for i in range(len(cl_sizes))] + ['red']
    all_sizes.plot(kind='bar', ax=axes[1], color=colors_bar, edgecolor='white', linewidth=1.5)
    axes[1].set_title('DBSCAN Cluster Sizes', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Count'); axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(SEG_OUT, "05_dbscan_analysis.png"), dpi=150, bbox_inches='tight')
plt.close()
print("   [CHART] 05_dbscan_analysis.png saved")


# ----------------------------------------------------------------------
#  ALGORITHM 4 : GAUSSIAN MIXTURE MODEL (GMM)
# ----------------------------------------------------------------------
print("\n" + "-" * 60)
print("  Algorithm 4 : GAUSSIAN MIXTURE MODEL (GMM)")
print("-" * 60)

bic_scores, aic_scores = [], []
gmm_range = range(2, 11)
for n_ in gmm_range:
    g_ = GaussianMixture(n_components=n_, random_state=42, n_init=5, covariance_type='full')
    g_.fit(X_seg)
    bic_scores.append(g_.bic(X_seg))
    aic_scores.append(g_.aic(X_seg))

opt_n = gmm_range[int(np.argmin(bic_scores))]
gmm = GaussianMixture(n_components=opt_n, random_state=42, n_init=5, covariance_type='full')
gmm_labels = gmm.fit_predict(X_seg)

sil_gmm = silhouette_score(X_seg, gmm_labels)
dbi_gmm = davies_bouldin_score(X_seg, gmm_labels)
chi_gmm = calinski_harabasz_score(X_seg, gmm_labels)

seg_results['GMM'] = {
    'clusters': opt_n, 'silhouette': sil_gmm,
    'davies_bouldin': dbi_gmm, 'calinski_harabasz': chi_gmm,
    'labels': gmm_labels
}

print(f"   Optimal components  : {opt_n}")
print(f"   Silhouette Score    : {sil_gmm:.4f}")
print(f"   Davies-Bouldin Index: {dbi_gmm:.4f}")
print(f"   Calinski-Harabasz   : {chi_gmm:.2f}")
print(f"   Log-Likelihood      : {gmm.score(X_seg):.4f}")

# GMM Visualization
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Gaussian Mixture Model (GMM) Analysis", fontsize=16, fontweight='bold')

# BIC + AIC
axes[0].plot(list(gmm_range), bic_scores, 'o-', color='#3498db', lw=2.5, label='BIC', markersize=8)
axes[0].plot(list(gmm_range), aic_scores, 's--', color='#e74c3c', lw=2.5, label='AIC', markersize=8)
axes[0].axvline(opt_n, color='#2ecc71', ls=':', lw=2, label=f'Optimal n={opt_n}')
axes[0].set_title('BIC & AIC vs Components', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Number of Components'); axes[0].set_ylabel('Score')
axes[0].legend(fontsize=11); axes[0].grid(True, alpha=0.3)

# PCA 2D Scatter
scatter_gmm = axes[1].scatter(X_seg_2D[:, 0], X_seg_2D[:, 1],
                                c=gmm_labels, cmap='tab10', s=15, alpha=0.6)
axes[1].set_title(f'GMM Clusters (n={opt_n}) - PCA Projection', fontsize=13, fontweight='bold')
axes[1].set_xlabel(f'PC1'); axes[1].set_ylabel(f'PC2')
axes[1].grid(True, alpha=0.3)
plt.colorbar(scatter_gmm, ax=axes[1], label='Cluster')

# Cluster distribution
cluster_counts_gmm = pd.Series(gmm_labels).value_counts().sort_index()
bars_gmm = axes[2].bar(cluster_counts_gmm.index, cluster_counts_gmm.values,
                         color=[COLORS[i % len(COLORS)] for i in range(len(cluster_counts_gmm))],
                         edgecolor='white', linewidth=1.5)
for bar, val in zip(bars_gmm, cluster_counts_gmm.values):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                  str(val), ha='center', fontweight='bold', fontsize=11)
axes[2].set_title('GMM Cluster Size Distribution', fontsize=13, fontweight='bold')
axes[2].set_xlabel('Cluster'); axes[2].set_ylabel('Count')
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(SEG_OUT, "06_gmm_analysis.png"), dpi=150, bbox_inches='tight')
plt.close()
print("   [CHART] 06_gmm_analysis.png saved")


# ----------------------------------------------------------------------
#  SEGMENTATION - COMPARISON SUMMARY
# ----------------------------------------------------------------------
print("\n" + "-" * 60)
print("  SEGMENTATION - ALGORITHM COMPARISON")
print("-" * 60)

seg_comparison = pd.DataFrame({
    'Algorithm': ['K-Means', 'Hierarchical', 'DBSCAN', 'GMM'],
    'Clusters': [seg_results[a]['clusters'] for a in ['K-Means', 'Hierarchical', 'DBSCAN', 'GMM']],
    'Silhouette ((up))': [seg_results[a]['silhouette'] for a in ['K-Means', 'Hierarchical', 'DBSCAN', 'GMM']],
    'Davies-Bouldin ((down))': [seg_results[a]['davies_bouldin'] for a in ['K-Means', 'Hierarchical', 'DBSCAN', 'GMM']],
    'Calinski-Harabasz ((up))': [seg_results[a]['calinski_harabasz'] for a in ['K-Means', 'Hierarchical', 'DBSCAN', 'GMM']],
}).set_index('Algorithm')
print(seg_comparison.round(4).to_string())

# Save comparison table
seg_comparison.round(4).to_csv(os.path.join(COMP_OUT, "segmentation_comparison.csv"))

# Comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Customer Segmentation - Algorithm Comparison", fontsize=16, fontweight='bold')

algos = seg_comparison.index.tolist()
x = np.arange(len(algos))
bar_colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

# Silhouette
bars = axes[0].bar(x, seg_comparison['Silhouette ((up))'], color=bar_colors, edgecolor='white', linewidth=1.5)
for bar in bars:
    axes[0].annotate(f'{bar.get_height():.4f}', (bar.get_x()+bar.get_width()/2, bar.get_height()),
                      ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[0].set_xticks(x); axes[0].set_xticklabels(algos, fontsize=11)
axes[0].set_title('Silhouette Score (Higher=Better)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Score'); axes[0].grid(True, alpha=0.3, axis='y')

# Davies-Bouldin
bars = axes[1].bar(x, seg_comparison['Davies-Bouldin ((down))'], color=bar_colors, edgecolor='white', linewidth=1.5)
for bar in bars:
    axes[1].annotate(f'{bar.get_height():.4f}', (bar.get_x()+bar.get_width()/2, bar.get_height()),
                      ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[1].set_xticks(x); axes[1].set_xticklabels(algos, fontsize=11)
axes[1].set_title('Davies-Bouldin Index (Lower=Better)', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Score'); axes[1].grid(True, alpha=0.3, axis='y')

# Calinski-Harabasz
bars = axes[2].bar(x, seg_comparison['Calinski-Harabasz ((up))'], color=bar_colors, edgecolor='white', linewidth=1.5)
for bar in bars:
    axes[2].annotate(f'{bar.get_height():.0f}', (bar.get_x()+bar.get_width()/2, bar.get_height()),
                      ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[2].set_xticks(x); axes[2].set_xticklabels(algos, fontsize=11)
axes[2].set_title('Calinski-Harabasz Score (Higher=Better)', fontsize=13, fontweight='bold')
axes[2].set_ylabel('Score'); axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(SEG_OUT, "07_segmentation_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("   [CHART] 07_segmentation_comparison.png saved")


# +===========================================================================+
# |           PART B : FRAUD DETECTION (Supervised / Anomaly)                |
# +===========================================================================+

print("\n" + "=" * 80)
print("  PART B : FRAUD DETECTION (Supervised + Anomaly Detection)")
print("=" * 80)

fraud_results = {}

# --- Helper function: evaluation plots ----------------------------------
def eval_fraud_model(name, y_true, y_pred, y_proba, save_dir):
    """Generate confusion matrix and ROC curve for a fraud detection model."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"{name} - Evaluation", fontsize=16, fontweight='bold')

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'],
                annot_kws={'size': 14})
    axes[0].set_title('Confusion Matrix', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Actual', fontsize=11); axes[0].set_xlabel('Predicted', fontsize=11)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    axes[1].plot(fpr, tpr, lw=2.5, color='#3498db', label=f'ROC-AUC = {auc:.4f}')
    axes[1].fill_between(fpr, tpr, alpha=0.15, color='#3498db')
    axes[1].plot([0,1], [0,1], 'k--', alpha=0.5)
    axes[1].set_title('ROC Curve', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('False Positive Rate', fontsize=11)
    axes[1].set_ylabel('True Positive Rate', fontsize=11)
    axes[1].legend(fontsize=12); axes[1].grid(True, alpha=0.3)

    # Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    axes[2].plot(rec, prec, lw=2.5, color='#e74c3c', label=f'AP = {ap:.4f}')
    axes[2].fill_between(rec, prec, alpha=0.15, color='#e74c3c')
    axes[2].set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Recall', fontsize=11); axes[2].set_ylabel('Precision', fontsize=11)
    axes[2].legend(fontsize=12); axes[2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fname = os.path.join(save_dir, f"{name.lower().replace(' ','_')}_eval.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   [CHART] {os.path.basename(fname)} saved")
    return auc, ap


# ----------------------------------------------------------------------
#  ALGORITHM 5 : LOGISTIC REGRESSION
# ----------------------------------------------------------------------
print("\n" + "-" * 60)
print("  Algorithm 5 : LOGISTIC REGRESSION")
print("-" * 60)

lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, C=1.0)
lr.fit(X_train_sc, y_train)
lr_pred  = lr.predict(X_test_sc)
lr_proba = lr.predict_proba(X_test_sc)[:, 1]

print(classification_report(y_test, lr_pred, target_names=['Legitimate', 'Fraud']))
auc_lr, ap_lr = eval_fraud_model("Logistic Regression", y_test, lr_pred, lr_proba, FRAUD_OUT)

fraud_results['Logistic Regression'] = {
    'accuracy': accuracy_score(y_test, lr_pred),
    'precision': precision_score(y_test, lr_pred),
    'recall': recall_score(y_test, lr_pred),
    'f1': f1_score(y_test, lr_pred),
    'auc': auc_lr,
    'ap': ap_lr,
    'proba': lr_proba
}


# ----------------------------------------------------------------------
#  ALGORITHM 6 : RANDOM FOREST
# ----------------------------------------------------------------------
print("\n" + "-" * 60)
print("  Algorithm 6 : RANDOM FOREST")
print("-" * 60)

rf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                             random_state=42, n_jobs=-1, max_depth=10)
rf.fit(X_train_sc, y_train)
rf_pred  = rf.predict(X_test_sc)
rf_proba = rf.predict_proba(X_test_sc)[:, 1]

print(classification_report(y_test, rf_pred, target_names=['Legitimate', 'Fraud']))
auc_rf, ap_rf = eval_fraud_model("Random Forest", y_test, rf_pred, rf_proba, FRAUD_OUT)

fraud_results['Random Forest'] = {
    'accuracy': accuracy_score(y_test, rf_pred),
    'precision': precision_score(y_test, rf_pred),
    'recall': recall_score(y_test, rf_pred),
    'f1': f1_score(y_test, rf_pred),
    'auc': auc_rf,
    'ap': ap_rf,
    'proba': rf_proba
}

# Feature importance
feat_names = [f'V{i}' for i in range(1, 29)] + ['Amount_Scaled', 'Time_Scaled']
feat_imp = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 8))
feat_imp[-15:].plot(kind='barh', ax=ax, color='#3498db', edgecolor='white', linewidth=1)
ax.set_title('Random Forest - Top 15 Feature Importances', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance', fontsize=12); ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(FRAUD_OUT, "rf_feature_importance.png"), dpi=150, bbox_inches='tight')
plt.close()
print("   [CHART] rf_feature_importance.png saved")


# ----------------------------------------------------------------------
#  ALGORITHM 7 : ISOLATION FOREST (Anomaly Detection)
# ----------------------------------------------------------------------
print("\n" + "-" * 60)
print("  Algorithm 7 : ISOLATION FOREST (Anomaly Detection)")
print("-" * 60)

contamination_rate = n_fraud_c / len(y_fraud)
iso = IsolationForest(n_estimators=200, contamination=contamination_rate,
                       random_state=42, n_jobs=-1)
iso.fit(X_train_sc)
iso_raw = iso.predict(X_test_sc)
iso_pred = (iso_raw == -1).astype(int)       # -1 = anomaly -> 1 = fraud
iso_scores = -iso.score_samples(X_test_sc)   # Higher = more anomalous

print(classification_report(y_test, iso_pred, target_names=['Legitimate', 'Fraud']))
auc_iso, ap_iso = eval_fraud_model("Isolation Forest", y_test, iso_pred, iso_scores, FRAUD_OUT)

fraud_results['Isolation Forest'] = {
    'accuracy': accuracy_score(y_test, iso_pred),
    'precision': precision_score(y_test, iso_pred),
    'recall': recall_score(y_test, iso_pred),
    'f1': f1_score(y_test, iso_pred),
    'auc': auc_iso,
    'ap': ap_iso,
    'proba': iso_scores
}


# ----------------------------------------------------------------------
#  ALGORITHM 8 : XGBOOST
# ----------------------------------------------------------------------
print("\n" + "-" * 60)
print("  Algorithm 8 : XGBOOST")
print("-" * 60)

scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
xgb = XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    scale_pos_weight=scale_pos_weight, random_state=42,
    eval_metric='logloss', verbosity=0, use_label_encoder=False
)
xgb.fit(X_train_sc, y_train)
xgb_pred  = xgb.predict(X_test_sc)
xgb_proba = xgb.predict_proba(X_test_sc)[:, 1]

print(classification_report(y_test, xgb_pred, target_names=['Legitimate', 'Fraud']))
auc_xgb, ap_xgb = eval_fraud_model("XGBoost", y_test, xgb_pred, xgb_proba, FRAUD_OUT)

fraud_results['XGBoost'] = {
    'accuracy': accuracy_score(y_test, xgb_pred),
    'precision': precision_score(y_test, xgb_pred),
    'recall': recall_score(y_test, xgb_pred),
    'f1': f1_score(y_test, xgb_pred),
    'auc': auc_xgb,
    'ap': ap_xgb,
    'proba': xgb_proba
}


# ----------------------------------------------------------------------
#  FRAUD DETECTION - COMPARISON SUMMARY
# ----------------------------------------------------------------------
print("\n" + "-" * 60)
print("  FRAUD DETECTION - ALGORITHM COMPARISON")
print("-" * 60)

fraud_comparison = pd.DataFrame({
    'Algorithm': list(fraud_results.keys()),
    'Accuracy': [fraud_results[a]['accuracy'] for a in fraud_results],
    'Precision': [fraud_results[a]['precision'] for a in fraud_results],
    'Recall': [fraud_results[a]['recall'] for a in fraud_results],
    'F1-Score': [fraud_results[a]['f1'] for a in fraud_results],
    'ROC-AUC': [fraud_results[a]['auc'] for a in fraud_results],
    'Avg Precision': [fraud_results[a]['ap'] for a in fraud_results],
}).set_index('Algorithm')
print(fraud_comparison.round(4).to_string())

fraud_comparison.round(4).to_csv(os.path.join(COMP_OUT, "fraud_comparison.csv"))

# Combined ROC Curves
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Fraud Detection - Algorithm Comparison", fontsize=16, fontweight='bold')

roc_colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
for (name, res), col in zip(fraud_results.items(), roc_colors):
    fpr, tpr, _ = roc_curve(y_test, res['proba'])
    axes[0].plot(fpr, tpr, lw=2.5, color=col, label=f"{name} (AUC={res['auc']:.4f})")
axes[0].plot([0,1], [0,1], 'k--', alpha=0.5)
axes[0].set_title('ROC Curves - All Algorithms', fontsize=14, fontweight='bold')
axes[0].set_xlabel('False Positive Rate', fontsize=12)
axes[0].set_ylabel('True Positive Rate', fontsize=12)
axes[0].legend(fontsize=11); axes[0].grid(True, alpha=0.3)

# Bar comparison
metrics_to_plot = ['F1-Score', 'ROC-AUC', 'Avg Precision']
x = np.arange(len(fraud_comparison))
width = 0.25
for i, metric in enumerate(metrics_to_plot):
    bars = axes[1].bar(x + i*width, fraud_comparison[metric],
                        width, label=metric, color=roc_colors[i], edgecolor='white', linewidth=1)
    for bar in bars:
        axes[1].annotate(f'{bar.get_height():.3f}',
                          (bar.get_x() + bar.get_width()/2, bar.get_height()),
                          ha='center', va='bottom', fontsize=8, fontweight='bold')
axes[1].set_xticks(x + width)
axes[1].set_xticklabels(fraud_comparison.index, fontsize=10, rotation=10)
axes[1].set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Score'); axes[1].legend(fontsize=10); axes[1].grid(True, alpha=0.3, axis='y')
axes[1].set_ylim(0, 1.15)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(FRAUD_OUT, "fraud_comparison_all.png"), dpi=150, bbox_inches='tight')
plt.close()
print("   [CHART] fraud_comparison_all.png saved")


# +===========================================================================+
# |               SECTION 6 : OVERALL PROJECT SUMMARY                       |
# +===========================================================================+

print("\n" + "=" * 80)
print("  OVERALL PROJECT SUMMARY")
print("=" * 80)

# Grand comparison chart
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle("Mini Project - Complete Results Overview",
             fontsize=18, fontweight='bold', y=1.02)

# Part A: Segmentation
seg_algos = ['K-Means', 'Hierarchical', 'DBSCAN', 'GMM']
sil_scores = [seg_results[a]['silhouette'] for a in seg_algos]
x_seg = np.arange(len(seg_algos))
bars_a = axes[0].bar(x_seg, sil_scores, color=bar_colors, edgecolor='white',
                      linewidth=2, width=0.6)
for bar in bars_a:
    axes[0].annotate(f'{bar.get_height():.4f}',
                      (bar.get_x() + bar.get_width()/2, bar.get_height()),
                      ha='center', va='bottom', fontsize=11, fontweight='bold')
axes[0].set_xticks(x_seg); axes[0].set_xticklabels(seg_algos, fontsize=12)
axes[0].set_title('Part A: Customer Segmentation\n(Silhouette Score - Higher is Better)',
                   fontsize=14, fontweight='bold')
axes[0].set_ylabel('Silhouette Score', fontsize=12)
axes[0].grid(True, alpha=0.3, axis='y')

# Part B: Fraud Detection
fraud_algos = list(fraud_results.keys())
auc_scores = [fraud_results[a]['auc'] for a in fraud_algos]
x_fraud = np.arange(len(fraud_algos))
bars_b = axes[1].bar(x_fraud, auc_scores, color=bar_colors, edgecolor='white',
                      linewidth=2, width=0.6)
for bar in bars_b:
    axes[1].annotate(f'{bar.get_height():.4f}',
                      (bar.get_x() + bar.get_width()/2, bar.get_height()),
                      ha='center', va='bottom', fontsize=11, fontweight='bold')
axes[1].set_xticks(x_fraud); axes[1].set_xticklabels(fraud_algos, fontsize=10, rotation=10)
axes[1].set_title('Part B: Fraud Detection\n(ROC-AUC - Higher is Better)',
                   fontsize=14, fontweight='bold')
axes[1].set_ylabel('ROC-AUC', fontsize=12)
axes[1].set_ylim(0, 1.1)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(COMP_OUT, "overall_project_summary.png"), dpi=150, bbox_inches='tight')
plt.close()
print("   [CHART] overall_project_summary.png saved")

# Print final summary
print("\n+---------------------------------------------------------------------+")
print("|                    FINAL RESULTS SUMMARY                           |")
print("+---------------------------------------------------------------------|")
print("|  PART A - CUSTOMER SEGMENTATION                                    |")
print("|  ---------------------------------                                 |")
best_seg = max(seg_results.items(), key=lambda x: x[1]['silhouette'])
print(f"|  Best Algorithm: {best_seg[0]:<20} Silhouette: {best_seg[1]['silhouette']:.4f}         |")
for algo, res in seg_results.items():
    print(f"|    {algo:<15} -> Sil: {res['silhouette']:.4f}  DBI: {res['davies_bouldin']:.4f}  CHI: {res['calinski_harabasz']:.0f}    |")
print("|                                                                     |")
print("|  PART B - FRAUD DETECTION                                          |")
print("|  ------------------------                                          |")
best_fraud = max(fraud_results.items(), key=lambda x: x[1]['auc'])
print(f"|  Best Algorithm: {best_fraud[0]:<20} AUC: {best_fraud[1]['auc']:.4f}              |")
for algo, res in fraud_results.items():
    print(f"|    {algo:<20} -> AUC: {res['auc']:.4f}  F1: {res['f1']:.4f}  AP: {res['ap']:.4f}   |")
print("+---------------------------------------------------------------------+")

print("\n" + "=" * 80)
print("  [DONE] PROJECT COMPLETE!")
print(f"  [DIR] Segmentation outputs : {SEG_OUT}")
print(f"  [DIR] Fraud detection outputs : {FRAUD_OUT}")
print(f"  [DIR] Comparison outputs  : {COMP_OUT}")
print(f"  [DIR] Datasets saved to   : {DATA_DIR}")
print("=" * 80)

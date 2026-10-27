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
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.colors import ListedColormap

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

# -- Journal-ready figure settings ---------------------------------------------
# Every figure is drawn at the physical size it occupies in the IEEE paper
# (3.5 in = one column, 7.16 in = full text width) with ~8 pt text, so the PNGs
# are placed at 100 % scale (no shrinking) and stay readable at default PDF zoom.
COL_W, FULL_W = 3.5, 7.16

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8,
    'axes.titlesize': 8.5, 'axes.titleweight': 'bold', 'axes.titlepad': 4,
    'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'legend.fontsize': 7, 'legend.framealpha': 0.9, 'legend.borderpad': 0.3,
    'axes.linewidth': 0.6, 'grid.linewidth': 0.4, 'lines.linewidth': 1.5,
    'lines.markersize': 4, 'figure.dpi': 100, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.04,
})

# Short, readable feature names for axes/titles
FEATURE_LABELS = {
    'BALANCE': 'Balance', 'BALANCE_FREQUENCY': 'Balance freq.', 'PURCHASES': 'Purchases',
    'ONEOFF_PURCHASES': 'One-off purch.', 'INSTALLMENTS_PURCHASES': 'Installment purch.',
    'CASH_ADVANCE': 'Cash advance', 'PURCHASES_FREQUENCY': 'Purchase freq.',
    'ONEOFF_PURCHASES_FREQUENCY': 'One-off freq.',
    'PURCHASES_INSTALLMENTS_FREQUENCY': 'Installment freq.',
    'CASH_ADVANCE_FREQUENCY': 'Cash adv. freq.', 'CASH_ADVANCE_TRX': 'Cash adv. trx',
    'PURCHASES_TRX': 'Purchase trx', 'CREDIT_LIMIT': 'Credit limit', 'PAYMENTS': 'Payments',
    'MINIMUM_PAYMENTS': 'Min. payments', 'PRC_FULL_PAYMENT': 'Full-payment ratio',
    'TENURE': 'Tenure',
}


def save_fig(fig, path):
    """Tight layout + save at 300 dpi; no text scaling is applied afterwards."""
    fig.tight_layout(pad=0.4, w_pad=0.9, h_pad=0.9)
    fig.savefig(path)
    plt.close(fig)


def kfmt(v, pos=None):
    """Compact tick labels: 6000 -> 6k, 12500 -> 12.5k."""
    if abs(v) >= 1000:
        return f'{v / 1000:g}k'
    return f'{v:g}'


def label_bars(ax, bars, fmt='{:.3f}', rotation=0, pad=2, size=None):
    """Write each bar's value just above the bar."""
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                    xytext=(0, pad), textcoords='offset points', ha='center',
                    va='bottom', rotation=rotation, fontsize=size)


def headroom(ax, frac=0.18):
    """Add empty space above the tallest bar so value labels never touch the title."""
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * frac)


def legend_below(ax, ncol, **kw):
    """Place a legend under the axes so it never covers the data."""
    leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=ncol, frameon=False,
                    handletextpad=0.2, columnspacing=0.9, **kw)
    handles = getattr(leg, 'legend_handles', None) or getattr(leg, 'legendHandles', [])
    for h in handles:                       # uniform, readable legend markers
        if hasattr(h, 'set_sizes'):
            h.set_sizes([22])
        h.set_alpha(1)
    return leg


def ygrid(ax):
    """Light horizontal grid only (no vertical lines cutting through bars)."""
    ax.grid(False)
    ax.grid(True, axis='y', alpha=0.3)


def wrap_names(names):
    return [n.replace(' ', '\n', 1) if len(n) > 9 else n for n in names]


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
df_cc = pd.read_csv(os.path.join(DATA_DIR, "dataset1_cc_general.csv"))
print(f"   Loaded from local file  ->  {df_cc.shape}")

# --- Dataset 2: Credit Card Fraud Detection (Kaggle - ULB) --------------
# Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# 284,807 transactions, PCA features V1-V28 + Time + Amount + Class
print("\n[Dataset 2] Credit Card Fraud Detection (European Transactions)")
df_fraud_raw = pd.read_csv(os.path.join(DATA_DIR, "dataset2_fraud_detection.csv"), nrows=20000)
print(f"   Loaded from local file  ->  {df_fraud_raw.shape}")

# --- Dataset 3: Default of Credit Card Clients (UCI ML Repository) ------
# Source: https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
# 30,000 observations with demographics + payment history
print("\n[Dataset 3] Default of Credit Card Clients (Taiwan)")
df_default = pd.read_csv(os.path.join(DATA_DIR, "dataset3_default_clients.csv"), nrows=15000)
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

fig, axes = plt.subplots(2, 3, figsize=(FULL_W, 3.7))
num_cols_cc = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS', 'TENURE']
for i, col in enumerate(num_cols_cc):
    ax = axes[i // 3][i % 3]
    if col in df_cc.columns:
        ax.hist(df_cc[col].dropna(), bins=40, color=COLORS[i], alpha=0.85,
                edgecolor='white', linewidth=0.3)
        ax.set_title(FEATURE_LABELS.get(col, col))
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
        ax.yaxis.set_major_formatter(FuncFormatter(kfmt))
        if i % 3 == 0:
            ax.set_ylabel('Frequency')
        if i // 3 == 1:
            ax.set_xlabel('Value')
save_fig(fig, os.path.join(SEG_OUT, "01_eda_cc_general.png"))
print("   [CHART] 01_eda_cc_general.png saved")

# Correlation heatmap
fig, ax = plt.subplots(figsize=(FULL_W, 5.6))
numeric_cc = df_cc.select_dtypes(include=[np.number])
corr = numeric_cc.corr().rename(index=FEATURE_LABELS, columns=FEATURE_LABELS)
corr = corr.iloc[1:, :-1]                         # drop the empty first row / last column
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r', vmin=-1, vmax=1,
            ax=ax, linewidths=0.4, linecolor='white', annot_kws={'size': 6.5},
            cbar_kws={'shrink': 0.8, 'pad': 0.02, 'label': 'Pearson correlation'})
ax.grid(False)
ax.tick_params(axis='x', labelsize=7, rotation=45)
ax.tick_params(axis='y', labelsize=7, rotation=0)
plt.setp(ax.get_xticklabels(), ha='right', rotation_mode='anchor')
save_fig(fig, os.path.join(SEG_OUT, "02_correlation_heatmap.png"))
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
fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 5.4))

# Elbow Method
ax = axes[0][0]
ax.plot(list(K_range), inertias, 'o-', color='#3498db')
ax.axvline(opt_k, color='#e74c3c', ls='--', lw=1.2, label=f'k = {opt_k}')
ax.set_title('Elbow method')
ax.set_xlabel('Number of clusters (k)'); ax.set_ylabel('Inertia (WCSS)')
ax.yaxis.set_major_formatter(FuncFormatter(kfmt))
ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)

# Silhouette Scores
ax = axes[0][1]
ax.plot(list(K_range), sil_km, 's-', color='#2ecc71')
ax.axvline(opt_k, color='#e74c3c', ls='--', lw=1.2, label=f'Best k = {opt_k}')
ax.annotate(f'{max(sil_km):.4f}', (opt_k, max(sil_km)), xytext=(6, 4),
            textcoords='offset points', fontweight='bold')
ax.set_title('Silhouette score vs. k')
ax.set_xlabel('Number of clusters (k)'); ax.set_ylabel('Silhouette score')
ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)
headroom(ax, 0.12)

# PCA 2D Scatter
ax = axes[1][0]
for c in range(opt_k):
    m = km_labels == c
    ax.scatter(X_seg_2D[m, 0], X_seg_2D[m, 1], s=2.5, alpha=0.55, linewidths=0,
               color=COLORS[c % len(COLORS)], label=f'Cluster {c}')
centers_2D = pca_seg.transform(km.cluster_centers_)
ax.scatter(centers_2D[:, 0], centers_2D[:, 1], c='black', marker='X', s=50,
           edgecolors='white', linewidths=0.8, label='Centroid', zorder=5)
ax.set_title(f'K-Means clusters (k = {opt_k}), PCA projection')
ax.set_xlabel(f'PC1 ({pca_seg.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca_seg.explained_variance_ratio_[1]*100:.1f}%)')
ax.grid(True, alpha=0.3)
legend_below(ax, ncol=opt_k + 1)

# Cluster Distribution
ax = axes[1][1]
cluster_counts = pd.Series(km_labels).value_counts().sort_index()
bars = ax.bar(cluster_counts.index, cluster_counts.values,
              color=[COLORS[i % len(COLORS)] for i in cluster_counts.index],
              edgecolor='white', linewidth=0.8, width=0.7)
for b, val in zip(bars, cluster_counts.values):
    ax.annotate(f'{val:,}\n({val / len(km_labels) * 100:.1f}%)',
                (b.get_x() + b.get_width() / 2, val), xytext=(0, 2),
                textcoords='offset points', ha='center', va='bottom', fontweight='bold')
headroom(ax, 0.28)
ax.set_xticks(list(cluster_counts.index))
ax.set_title('Cluster size distribution')
ax.set_xlabel('Cluster'); ax.set_ylabel('Customers')
ygrid(ax)

save_fig(fig, os.path.join(SEG_OUT, "03_kmeans_analysis.png"))
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
fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.1))

# Dendrogram (on a reproducible random sample for readability)
rng_dendro = np.random.RandomState(42)
sample_idx = rng_dendro.choice(len(X_seg), min(80, len(X_seg)), replace=False)
Z = linkage(X_seg[sample_idx], method='ward')
cut_height = Z[-opt_k + 1, 2]
dendrogram(Z, ax=axes[0], no_labels=True, color_threshold=cut_height,
           above_threshold_color='gray')
axes[0].axhline(cut_height, color='red', ls='--', lw=1.2, label=f'Cut at k = {opt_k}')
axes[0].set_title(f'Dendrogram (Ward, {len(sample_idx)} sampled customers)')
axes[0].set_xlabel('Sampled customers'); axes[0].set_ylabel('Ward distance')
axes[0].legend(loc='upper right'); axes[0].grid(False)

# PCA 2D Scatter
ax = axes[1]
for c in range(opt_k):
    m = hc_labels == c
    ax.scatter(X_seg_2D[m, 0], X_seg_2D[m, 1], s=2.5, alpha=0.55, linewidths=0,
               color=COLORS[c % len(COLORS)], label=f'Cluster {c} (n={m.sum():,})')
ax.set_title(f'Hierarchical clusters (k = {opt_k}), PCA projection')
ax.set_xlabel(f'PC1 ({pca_seg.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca_seg.explained_variance_ratio_[1]*100:.1f}%)')
ax.grid(True, alpha=0.3)
legend_below(ax, ncol=opt_k)

save_fig(fig, os.path.join(SEG_OUT, "04_hierarchical_analysis.png"))
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
fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.1))

# Cluster plot (noise drawn last so it stays visible)
ax = axes[0]
for label in sorted(l for l in set(db_labels) if l != -1):
    m = db_labels == label
    ax.scatter(X_seg_2D[m, 0], X_seg_2D[m, 1], c=COLORS[label % len(COLORS)],
               s=3 if label == 0 else 9, alpha=0.6, linewidths=0,
               label=f'C{label} ({m.sum():,})')
m_noise = db_labels == -1
ax.scatter(X_seg_2D[m_noise, 0], X_seg_2D[m_noise, 1], c='red', marker='x', s=6,
           alpha=0.6, linewidths=0.5, label=f'Noise ({m_noise.sum():,})', zorder=4)
ax.set_title(f'DBSCAN (eps = {best_eps}, MinPts = {best_ms})')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.grid(True, alpha=0.3)
legend_below(ax, ncol=3)

# Cluster size distribution
ax = axes[1]
cl_sizes = pd.Series(db_labels[db_labels != -1]).value_counts().sort_index()
names_db = [f'C{i}' for i in cl_sizes.index] + ['Noise']
vals_db = list(cl_sizes.values) + [nn_db]
cols_db = [COLORS[i % len(COLORS)] for i in cl_sizes.index] + ['red']
bars = ax.bar(names_db, vals_db, color=cols_db, edgecolor='white', linewidth=0.8, width=0.6)
for b, v in zip(bars, vals_db):
    ax.annotate(f'{v:,}', (b.get_x() + b.get_width() / 2, v), xytext=(0, 2),
                textcoords='offset points', ha='center', va='bottom', fontweight='bold')
headroom(ax, 0.12)
ax.set_title('DBSCAN cluster sizes'); ax.set_ylabel('Customers')
ygrid(ax)

save_fig(fig, os.path.join(SEG_OUT, "05_dbscan_analysis.png"))
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
fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.8))

# BIC + AIC
ax = axes[0]
ax.plot(list(gmm_range), bic_scores, 'o-', color='#3498db', label='BIC')
ax.plot(list(gmm_range), aic_scores, 's--', color='#e74c3c', label='AIC')
ax.axvline(opt_n, color='#2ecc71', ls=':', lw=1.5, label=f'n = {opt_n}')
ax.set_title('BIC and AIC')
ax.set_xlabel('Components'); ax.set_ylabel('Information criterion')
ax.yaxis.set_major_formatter(FuncFormatter(kfmt))
ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)

# PCA 2D Scatter with a discrete colour bar
ax = axes[1]
cmap_gmm = ListedColormap([COLORS[i % len(COLORS)] for i in range(opt_n)])
sc = ax.scatter(X_seg_2D[:, 0], X_seg_2D[:, 1], c=gmm_labels, cmap=cmap_gmm,
                vmin=-0.5, vmax=opt_n - 0.5, s=2.5, alpha=0.6, linewidths=0)
cb = fig.colorbar(sc, ax=ax, ticks=list(range(opt_n)), fraction=0.06, pad=0.02)
cb.set_label('Cluster'); cb.ax.tick_params(labelsize=6.5)
cb.solids.set_alpha(1)
ax.set_title(f'GMM clusters (n = {opt_n}), PCA')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.grid(True, alpha=0.3)

# Cluster distribution (counts written vertically so neighbours never collide)
ax = axes[2]
cluster_counts_gmm = pd.Series(gmm_labels).value_counts().sort_index()
bars = ax.bar(cluster_counts_gmm.index, cluster_counts_gmm.values,
              color=[COLORS[i % len(COLORS)] for i in cluster_counts_gmm.index],
              edgecolor='white', linewidth=0.8, width=0.75)
label_bars(ax, bars, fmt='{:,.0f}', rotation=90, size=6.5)
headroom(ax, 0.3)
ax.set_xticks(list(cluster_counts_gmm.index))
ax.set_title('GMM cluster sizes'); ax.set_xlabel('Cluster'); ax.set_ylabel('Customers')
ygrid(ax)

save_fig(fig, os.path.join(SEG_OUT, "06_gmm_analysis.png"))
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
fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.7))

algos = seg_comparison.index.tolist()
x = np.arange(len(algos))
bar_colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
panels = [('Silhouette ((up))', 'Silhouette (higher is better)', '{:.4f}'),
          ('Davies-Bouldin ((down))', 'Davies-Bouldin (lower is better)', '{:.3f}'),
          ('Calinski-Harabasz ((up))', 'Calinski-Harabasz (higher is better)', '{:,.0f}')]
short_titles = ['Silhouette \u2191', 'Davies-Bouldin \u2193', 'Calinski-Harabasz \u2191']
for ax, (col, _, fmt), ttl in zip(axes, panels, short_titles):
    bars = ax.bar(x, seg_comparison[col], color=bar_colors, edgecolor='white',
                  linewidth=0.8, width=0.7)
    label_bars(ax, bars, fmt=fmt, size=7)
    headroom(ax, 0.14)
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=25, ha='right', rotation_mode='anchor')
    ax.set_title(ttl); ygrid(ax)
axes[0].set_ylabel('Score')

save_fig(fig, os.path.join(SEG_OUT, "07_segmentation_comparison.png"))
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
    """Generate confusion matrix, ROC and precision-recall curves for one model."""
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.6))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,
                xticklabels=['Legitimate', 'Fraud'], yticklabels=['Legitimate', 'Fraud'],
                annot_kws={'size': 10, 'weight': 'bold'}, linewidths=0.5, linecolor='white')
    axes[0].set_title('Confusion matrix')
    axes[0].set_ylabel('Actual'); axes[0].set_xlabel('Predicted')
    plt.setp(axes[0].get_yticklabels(), rotation=90, va='center')

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    axes[1].plot(fpr, tpr, color='#3498db', label=f'AUC = {auc:.4f}')
    axes[1].fill_between(fpr, tpr, alpha=0.15, color='#3498db')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
    axes[1].set_title('ROC curve')
    axes[1].set_xlabel('False positive rate'); axes[1].set_ylabel('True positive rate')
    axes[1].legend(loc='lower right'); axes[1].grid(True, alpha=0.3)

    # Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    axes[2].plot(rec, prec, color='#e74c3c', label=f'AP = {ap:.4f}')
    axes[2].fill_between(rec, prec, alpha=0.15, color='#e74c3c')
    axes[2].set_title('Precision-recall curve')
    axes[2].set_xlabel('Recall'); axes[2].set_ylabel('Precision')
    axes[2].set_ylim(0, 1.05)
    axes[2].legend(loc='lower left'); axes[2].grid(True, alpha=0.3)

    fname = os.path.join(save_dir, f"{name.lower().replace(' ','_')}_eval.png")
    save_fig(fig, fname)
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
fig, ax = plt.subplots(figsize=(COL_W, 3.4))
top15 = feat_imp[-15:]
bars = ax.barh(top15.index, top15.values, color='#3498db', edgecolor='white', linewidth=0.4)
for b, v in zip(bars, top15.values):
    ax.annotate(f'{v:.3f}', (v, b.get_y() + b.get_height() / 2), xytext=(3, 0),
                textcoords='offset points', va='center', fontsize=6.5)
ax.set_xlim(0, top15.max() * 1.2)
ax.set_title('Random Forest: top 15 feature importances')
ax.set_xlabel('Gini importance'); ax.grid(False); ax.grid(True, axis='x', alpha=0.3)
save_fig(fig, os.path.join(FRAUD_OUT, "rf_feature_importance.png"))
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
fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.2))

roc_colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
for (name, res), col in zip(fraud_results.items(), roc_colors):
    fpr, tpr, _ = roc_curve(y_test, res['proba'])
    axes[0].plot(fpr, tpr, color=col, label=f"{name} ({res['auc']:.4f})")
axes[0].plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
axes[0].set_title('ROC curves (AUC in brackets)')
axes[0].set_xlabel('False positive rate'); axes[0].set_ylabel('True positive rate')
axes[0].legend(loc='lower right'); axes[0].grid(True, alpha=0.3)

# Grouped bar comparison (values written vertically above each bar)
metrics_to_plot = ['F1-Score', 'ROC-AUC', 'Avg Precision']
x = np.arange(len(fraud_comparison))
width = 0.26
for i, metric in enumerate(metrics_to_plot):
    bars = axes[1].bar(x + (i - 1) * width, fraud_comparison[metric], width,
                       label=metric, color=roc_colors[i], edgecolor='white', linewidth=0.6)
    label_bars(axes[1], bars, fmt='{:.3f}', rotation=90, size=6.5)
axes[1].set_xticks(x)
axes[1].set_xticklabels(wrap_names(fraud_comparison.index.tolist()))
axes[1].set_title('Performance metrics')
axes[1].set_ylabel('Score'); axes[1].set_ylim(0, 1.4)
axes[1].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
axes[1].legend(loc='upper center', ncol=3, fontsize=6.5)
ygrid(axes[1])

save_fig(fig, os.path.join(FRAUD_OUT, "fraud_comparison_all.png"))
print("   [CHART] fraud_comparison_all.png saved")


# +===========================================================================+
# |               SECTION 6 : OVERALL PROJECT SUMMARY                       |
# +===========================================================================+

print("\n" + "=" * 80)
print("  OVERALL PROJECT SUMMARY")
print("=" * 80)

# Grand comparison chart
fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.0))

# Part A: Segmentation
seg_algos = ['K-Means', 'Hierarchical', 'DBSCAN', 'GMM']
sil_scores = [seg_results[a]['silhouette'] for a in seg_algos]
x_seg = np.arange(len(seg_algos))
bars_a = axes[0].bar(x_seg, sil_scores, color=bar_colors, edgecolor='white',
                     linewidth=0.8, width=0.65)
label_bars(axes[0], bars_a, fmt='{:.4f}', size=7.5)
headroom(axes[0], 0.14)
axes[0].set_xticks(x_seg)
axes[0].set_xticklabels(wrap_names(seg_algos))
axes[0].set_title('Part A: Segmentation (Silhouette \u2191)')
axes[0].set_ylabel('Silhouette score')
ygrid(axes[0])

# Part B: Fraud Detection
fraud_algos = list(fraud_results.keys())
auc_scores = [fraud_results[a]['auc'] for a in fraud_algos]
x_fraud = np.arange(len(fraud_algos))
bars_b = axes[1].bar(x_fraud, auc_scores, color=bar_colors, edgecolor='white',
                     linewidth=0.8, width=0.65)
label_bars(axes[1], bars_b, fmt='{:.4f}', size=7.5)
axes[1].set_xticks(x_fraud)
axes[1].set_xticklabels(wrap_names(fraud_algos))
axes[1].set_title('Part B: Fraud detection (ROC-AUC \u2191)')
axes[1].set_ylabel('ROC-AUC')
axes[1].set_ylim(0, 1.12)
ygrid(axes[1])

save_fig(fig, os.path.join(COMP_OUT, "overall_project_summary.png"))
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


#  Dataset Description - Customer Segmentation and Clustering in Banking
#  Using Data Mining Techniques

## Datasets Used

This project utilizes **3 real-world datasets** sourced from reputable 
online platforms (Kaggle and UCI Machine Learning Repository). These 
datasets are combined and preprocessed to support both **Customer 
Segmentation** (unsupervised learning) and **Fraud Detection** 
(supervised + anomaly detection) tasks.

---

### Dataset 1: Credit Card Customer Behavioral Data (CC GENERAL)

| Attribute         | Detail                                              |
|-------------------|-----------------------------------------------------|
| **Source**         | Kaggle — arjunbhasin2013/ccdata                     |
| **URL**            | https://www.kaggle.com/datasets/arjunbhasin2013/ccdata |
| **Records**        | ~8,950 active credit card holders                   |
| **Features**       | 18 behavioral variables                             |
| **Time Period**    | 6 months of usage behavior                          |
| **Used For**       | **Customer Segmentation (Part A)**                  |

**Key Features:**
- `BALANCE` — Average monthly balance
- `PURCHASES` — Total purchases made
- `CASH_ADVANCE` — Cash advance amount
- `CREDIT_LIMIT` — Credit limit assigned
- `PAYMENTS` — Total payments made
- `PURCHASES_FREQUENCY` — Frequency of purchases (0–1)
- `CASH_ADVANCE_FREQUENCY` — Frequency of cash advances
- `TENURE` — Number of months as customer

**Why this dataset?**  
It provides rich behavioral signals for unsupervised clustering, enabling 
identification of distinct customer segments based on spending patterns, 
payment behaviors, and financial engagement levels.

---

### Dataset 2: Credit Card Fraud Detection (European Transactions)

| Attribute         | Detail                                              |
|-------------------|-----------------------------------------------------|
| **Source**         | Kaggle — mlg-ulb/creditcardfraud                    |
| **URL**            | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| **Records**        | 284,807 transactions (sampled 20,000 for efficiency)|
| **Features**       | 30 numerical features (PCA-transformed V1–V28, Time, Amount) |
| **Class Balance**  | Highly imbalanced (~0.17% fraud)                    |
| **Used For**       | **Fraud Detection (Part B)**                        |

**Key Features:**
- `V1` to `V28` — PCA-transformed features (anonymized for confidentiality)
- `Time` — Seconds elapsed since first transaction
- `Amount` — Transaction amount
- `Class` — Target variable (0 = Legitimate, 1 = Fraud)

**Why this dataset?**  
It is the most widely-cited fraud detection benchmark, published by 
Worldline and the Machine Learning Group of Université Libre de 
Bruxelles (ULB). The PCA transformation preserves privacy while 
retaining discriminative patterns.

---

### Dataset 3: Default of Credit Card Clients (Taiwan)

| Attribute         | Detail                                              |
|-------------------|-----------------------------------------------------|
| **Source**         | UCI Machine Learning Repository                     |
| **URL**            | https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients |
| **Records**        | 30,000 observations (sampled 15,000)                |
| **Features**       | 24 features (demographics + payment history)        |
| **Class Balance**  | ~22% default rate                                   |
| **Used For**       | **Supplementary risk indicators**                   |

**Key Features:**
- `LIMIT_BAL` — Credit limit
- `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` — Demographics
- `PAY_0` to `PAY_6` — Repayment status (6 months history)
- `BILL_AMT1` to `BILL_AMT6` — Bill statement amounts
- `PAY_AMT1` to `PAY_AMT6` — Previous payment amounts
- `default.payment.next.month` — Target (0 = No default, 1 = Default)

**Why this dataset?**  
It adds demographic and temporal payment dimensions to the analysis, 
providing complementary features for understanding credit risk patterns 
alongside the primary fraud detection dataset.

---

## Data Preprocessing Steps

### For Customer Segmentation (Part A):
1. **Missing Value Handling** — Median imputation for numerical columns
2. **Outlier Treatment** — IQR-based capping (1.5 × IQR)
3. **Feature Scaling** — StandardScaler (zero mean, unit variance)
4. **Dimensionality Reduction** — PCA to 2D for visualization

### For Fraud Detection (Part B):
1. **Feature Engineering** — Amount and Time normalization via StandardScaler
2. **Class Imbalance Handling** — `class_weight='balanced'` / `scale_pos_weight`
3. **Train-Test Split** — 80/20 stratified split (preserving class ratio)
4. **Scaling** — StandardScaler applied to train/test separately

---

## Common Columns Across Datasets

While the three datasets originate from different sources, they share 
several conceptually aligned features that were harmonized during 
preprocessing:

| Concept              | Dataset 1 (CC GENERAL)     | Dataset 3 (Taiwan Default) |
|----------------------|----------------------------|---------------------------|
| Credit Limit         | `CREDIT_LIMIT`             | `LIMIT_BAL`               |
| Payment Behavior     | `PAYMENTS`                 | `PAY_AMT1–6`              |
| Balance/Bill         | `BALANCE`                  | `BILL_AMT1–6`             |
| Temporal             | `TENURE`                   | `PAY_0–6` (6 months)      |
| Purchase/Transaction | `PURCHASES`, `PURCHASES_TRX`| Transaction counts       |

---

## Ethical Considerations

- All datasets are publicly available and anonymized
- No personally identifiable information (PII) is present
- PCA transformation in Dataset 2 ensures transaction privacy
- Results are for academic/research purposes only

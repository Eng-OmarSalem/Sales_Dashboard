[README.md](https://github.com/user-attachments/files/30721209/README.md)
<div align="center">

# 📡 Telecom Customer Churn Prediction
<img width="790" height="490" alt="risk_tiers" src="https://github.com/user-attachments/assets/6e6a594d-79cd-454a-8ada-1bbc22c54223" /><img width="989" height="539" alt="model_comparison_auc" src="https://github.com/user-attachments/assets/7c82b735-7b9f-4029-954a-8b9a13bef801" />
<img width="1490" height="740" alt="key_features_boxplots" src="https://github.com/user-attachments/assets/029cf029-64eb-4d65-b58a-c665708aa18a" />
<img width="840" height="790" alt="feature_importance" src="https://github.com/user-attachments/assets/2a6c63c3-cb52-482c-a352-abc4f596c91f" />
<img width="1083" height="440" alt="churn_distribution" src="https://github.com/user-attachments/assets/63b087e0-63b1-4d37-829e-6af06d596120" />
<img width="790" height="490" alt="churn_by_equipment_age" src="https://github.com/user-attachments/assets/1a50eade-4b6d-4ee7-912b-00ce3d15431d" />

### Predicting customer churn with a stacked ensemble of gradient-boosted models

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2C8EBB?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=for-the-badge&logo=lightgbm&logoColor=white)](https://lightgbm.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?style=for-the-badge&logo=catboost&logoColor=black)](https://catboost.ai/)
[![scikit-learn](https://img.shields.io/badge/sikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)

</div>

---

## 📌 Overview

Customer churn is one of the most expensive problems in the telecom industry — acquiring a new subscriber costs far more than retaining an existing one. This project builds an **end-to-end machine learning pipeline** that predicts which customers are likely to churn *before* they leave, so retention teams can act early and target the right customers.

The pipeline combines heavy feature engineering with a **stacked ensemble of XGBoost, LightGBM, CatBoost, Extra Trees, and Logistic Regression**, optimally blended to squeeze out the best possible predictive performance.

> 🎯 **Final model performance: AUC-ROC = 0.7076** on a held-out test set of 20,000 unseen customers.

---

## 🗂️ Dataset

Two raw sources are merged on `Customer_ID` into a single modeling table:

| File | Description |
|---|---|
| `Client.csv` | Demographic & account-level attributes (credit class, handset, income, household, etc.) |
| `Record.csv` | Usage & billing behavior (minutes of use, revenue, dropped calls, overage, customer care calls, etc.) |

| Metric | Value |
|---|---|
| Total customers | **100,000** |
| Raw features | **100** |
| Churn rate | **49.56 %** (well-balanced target) |
| Final engineered features | **142** |
| Train / Test split | 80,000 / 20,000 |

---

## 🔍 Exploratory Data Analysis

**Churn is almost perfectly balanced**, which makes AUC-ROC and F1 reliable evaluation metrics without needing heavy resampling.

![Churn distribution](assets/churn_distribution.png)

**Equipment age is one of the strongest churn drivers** — customers who haven't upgraded their device in 3+ years churn at a noticeably higher rate than the overall average.

![Churn rate by equipment age](assets/churn_by_equipment_age.png)

Digging into usage patterns confirms that **tenure (`months`) and days-since-upgrade (`eqpdays`) show the clearest separation** between churners and non-churners, alongside revenue and minutes-of-use trends.

![Key features vs churn](assets/key_features_boxplots.png)

---

## ⚙️ Feature Engineering

On top of the raw 100 columns, the pipeline adds:

- 📈 **Trend features** — 3-month vs 6-month usage/revenue momentum (`mou_trend_3v6`, `rev_trend_3v6`)
- ☎️ **Behavioral ratios** — dropped-call ratio, overage ratio, active-subscriber ratio
- 👥 **Peer-group aggregates** — features benchmarked against similar customer segments (fit on train, safely mapped to test)
- 🎯 **K-Fold target encoding** — for high-cardinality categorical variables, leak-free via out-of-fold encoding
- 🧩 **K-Means distance features** — cluster-distance signals from core usage/trend metrics

This engineering pushes the feature space from 100 → **142 model-ready features**.

---

## 🤖 Modeling Approach

Five base models are trained and evaluated, then combined into a two-level stack plus a final optimized weighted blend:

| Model | AUC-ROC |
|---|---|
| Logistic Regression | baseline |
| Random Forest | baseline |
| CatBoost | 0.7045 |
| LightGBM | 0.7043 |
| XGBoost | 0.7052 |
| XGBoost (5-seed bag) | 0.7071 |
| Two-level Stacking | 0.7057 |
| **🏆 Final Optimized Blend** | **0.7076** |

The final prediction is a **weighted blend optimized via `scipy.optimize`**, dominated by the 5-seed-bagged XGBoost and pairwise feature-interaction models:

![Model comparison AUC-ROC](assets/model_comparison_auc.png)

**Top churn drivers** identified by the XGBoost feature-importance ranking:

![Top 20 feature importances](assets/feature_importance.png)

---

## 🚦 Business Output — Risk Tiers

Every customer is finally scored and bucketed into an actionable **risk tier**, so retention/marketing teams can prioritize outreach:

![Customer distribution by churn risk tier](assets/risk_tiers.png)

| Tier | Action |
|---|---|
| 🟢 Low Risk | No action needed |
| 🟡 Medium Risk | Monitor / light engagement |
| 🟠 High Risk | Proactive retention offer |
| 🔴 Critical Risk | Immediate intervention |

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Data:** pandas, numpy
- **Modeling:** XGBoost, LightGBM, CatBoost, scikit-learn, category_encoders
- **Optimization:** scipy (blend-weight optimization)
- **Visualization:** matplotlib, seaborn
- **Environment:** Jupyter / Google Colab

---

## 📁 Project Structure

```
telecom-churn-prediction/
├── Telecom_Churn_Model.ipynb        # Full pipeline: EDA → feature engineering → modeling → blending
├── Telecom_Churn_Power_Point.pptx   # Executive presentation of findings
├── Client.csv                       # Customer demographic & account data
├── Record.csv                       # Customer usage & billing behavior data
├── assets/                          # Charts used in this README
└── README.md
```

## ▶️ How to Run

```bash
git clone https://github.com/<your-username>/telecom-churn-prediction.git
cd telecom-churn-prediction

pip install -q xgboost lightgbm catboost category_encoders pandas numpy scikit-learn matplotlib seaborn scipy

jupyter notebook Telecom_Churn_Model.ipynb
```

---

## 📬 Contact

<div align="center">

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Eng.%20Omar%20Salem-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/eng-omarsalem)
[![Email](https://img.shields.io/badge/Email-salemomar676%40gmail.com-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:salemomar676@gmail.com)
[![Portfolio](https://img.shields.io/badge/Portfolio-View%20Proposal-FFB000?style=for-the-badge&logo=googledocs&logoColor=white)](https://gamma.app/docs/Copy-of-Brand-Partnership-Proposal-lrp9yrhau9gdpj1)

**⭐ If you found this project useful, consider giving it a star!**

</div>

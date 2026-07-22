# Problematic Internet Use: Machine Learning and Time-Series Analysis

End-to-end Data Mining project based on the **Child Mind Institute – Problematic Internet Use** dataset.

The project combines tabular data, wrist-worn accelerometer time series, machine learning and explainable AI to investigate the relationship between physical activity, fitness, sleep-related indicators and problematic Internet use among children and adolescents.

Problematic Internet use is represented through the **Severity Impairment Index (`sii`)**.

---

## Project Overview

The analysis is organised into four main modules:

1. **Data Understanding and Preparation**
2. **Advanced Preprocessing**
3. **Advanced Machine Learning and Explainability**
4. **Time-Series Analysis**

The project includes:

- exploratory data analysis;
- missing-value treatment;
- feature engineering;
- correlation-based feature selection;
- outlier detection;
- imbalanced learning;
- multiclass classification;
- advanced regression;
- model explainability with SHAP and LORE;
- time-series preprocessing;
- motif and discord discovery;
- time-series clustering;
- shapelet and ROCKET classification.

---

## Dataset

The project uses data from the Kaggle competition:

[Child Mind Institute – Problematic Internet Use](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/data)

The tabular dataset contains demographic, behavioural and physiological information, including:

- age and sex;
- daily Internet usage;
- physical measurements;
- cardiovascular fitness;
- body composition;
- physical activity questionnaires;
- sleep-disturbance indicators;
- Parent-Child Internet Addiction Test variables.

The original tabular dataset contains **8,460 observations and 82 attributes**.

The time-series dataset contains **4,437 ENMO signals**, each composed of **200 time steps**. ENMO, or Euclidean Norm Minus One, provides a univariate representation of overall movement intensity extracted from accelerometer data.

The datasets must be downloaded separately and placed in the appropriate local data folder.

---

## Research Tasks

### 1. Data Preparation

The preprocessing pipeline includes:

- numerical imputation using the median;
- categorical imputation using the mode;
- treatment of physiologically invalid values;
- logarithmic transformation of highly skewed variables;
- feature scaling;
- removal of strongly correlated variables;
- prevention of target leakage from PCIAT variables;
- creation of a custom binary target named `sedentary_social_withdrawal`.

The custom target identifies participants showing both low physical activity and signs of social withdrawal or problematic digital behaviour.

---

### 2. Outlier Detection

Nine anomaly-detection methods from different algorithmic families were compared:

- HBOS;
- Gaussian Mixture Model;
- Elliptic Envelope;
- k-Nearest Neighbours;
- Local Outlier Factor;
- CBLOF;
- ABOD;
- LODA;
- Isolation Forest.

Each method identified approximately the top 1% most anomalous observations.

A consensus rule classified an observation as an outlier when it was detected by at least four algorithms. This procedure identified **56 consensus outliers**, which were visualised through PCA and removed before the following analyses.

---

### 3. Imbalanced Learning

A binary classification task was defined using `sedentary_social_withdrawal` as the target.

The original class distribution was strongly imbalanced, with approximately 96% negative observations.

The following resampling methods were evaluated:

#### Undersampling

- Random Undersampling;
- Condensed Nearest Neighbour;
- Tomek Links;
- Edited Nearest Neighbours;
- Cluster Centroids.

#### Oversampling

- Random Oversampling;
- SMOTE;
- ADASYN.

The baseline Decision Tree achieved only **0.14 recall** for the positive class.

The combined **Random Undersampling + Random Oversampling** strategy increased positive-class recall to **0.77**, although precision remained low because of the large number of false positives.

---

### 4. Advanced Classification

The main supervised-learning task predicts the Severity Impairment Index.

Because the Severe class contained very few observations, the Moderate and Severe categories were merged. The resulting target contains three classes:

- `None`;
- `Mild`;
- `Moderate + Severe`.

The following models were trained and tuned:

- Logistic Regression;
- Linear Support Vector Machine;
- RBF Support Vector Machine;
- Multilayer Perceptron;
- Random Forest;
- Gradient Boosting.

Hyperparameter optimisation was performed using grid search or randomised search with stratified cross-validation.

### Classification Results

| Model | Main result |
|---|---:|
| Logistic Regression | Strong performance mainly on the majority class |
| Linear SVM | More balanced than Logistic Regression |
| RBF SVM | Captured non-linear patterns but showed limited improvement |
| Multilayer Perceptron | Moderate multiclass performance |
| Random Forest | Macro F1-score: **0.80** |
| Gradient Boosting | Accuracy: **0.92**, Macro F1-score: **0.87** |

Gradient Boosting obtained the strongest overall performance and achieved relatively balanced results across all three severity classes.

The results suggest that non-linear interactions among sleep, physical activity, fitness and body-composition variables are important for predicting problematic Internet-use severity.

---

### 5. Advanced Regression

A regression task was defined to predict participants' physical heart rate.

Two non-linear regression models were evaluated:

- Random Forest Regressor;
- XGBoost Regressor.

| Model | R² | RMSE | MAE |
|---|---:|---:|---:|
| Random Forest Regressor | 0.100 | 10.06 | 7.11 |
| Tuned XGBoost Regressor | 0.117 | 9.96 | 7.05 |

XGBoost produced a small improvement over Random Forest. However, both models had limited explanatory power, indicating that the available variables explain only a small proportion of individual heart-rate variability.

---

### 6. Explainable AI

The tuned Random Forest classifier was analysed using:

- **SHAP**, for global and local feature contributions;
- **LORE**, for local decision rules and counterfactual explanations.

The explainability analysis investigates:

- which variables have the strongest global influence;
- why individual observations receive a specific prediction;
- which feature changes could produce a different predicted class;
- whether incorrect predictions occur near ambiguous decision regions.

Sleep-related variables, particularly the standardised Sleep Disturbance Scale score, emerged as important predictors.

The explanations represent predictive associations and must not be interpreted as clinical rules or causal relationships.

---

## Time-Series Analysis

The time-series analysis focuses on the `enmo` accelerometer channel.

### Preprocessing

The pipeline includes:

- exploratory signal analysis;
- per-series z-score standardisation;
- Hampel-filter anomaly checking;
- stationarity testing with the Augmented Dickey-Fuller test;
- detrending;
- deseasonalisation;
- Symbolic Aggregate approXimation.

The average profiles showed that problematic subjects had lower mean movement intensity. However, individual signals remained noisy and strongly overlapping.

---

### Motifs and Discords

The Matrix Profile framework was used to identify:

- **motifs**: recurring subsequences with low nearest-neighbour distance;
- **discords**: unusual subsequences with high nearest-neighbour distance.

Flat low-activity subsequences were filtered to prevent inactive periods from dominating motif discovery.

Motifs mainly represented recurring activity peaks, while discords represented isolated or irregular movement bursts.

---

### Time-Series Clustering

The following clustering methods were applied:

- Time Series K-means with Euclidean distance;
- Time Series K-means with Dynamic Time Warping;
- Agglomerative hierarchical clustering with Ward linkage.

Seven clusters were selected to provide interpretable activity profiles.

The clusters mainly differed in the timing and intensity of movement peaks. However, they did not clearly separate problematic from non-problematic participants:

- silhouette score: approximately **0.031**;
- cluster purity: **0.666**, close to the majority-class proportion.

This suggests that activity timing contains temporal structure but is not sufficient by itself to identify problematic Internet use.

---

### Time-Series Classification

Four classification approaches were compared:

| Model | Accuracy | Macro F1 | AUC |
|---|---:|---:|---:|
| KNN with Euclidean distance | 0.61 | 0.48 | 0.546 |
| KNN with DTW | 0.61 | 0.45 | 0.521 |
| Shapelet Transform + Logistic Regression | 0.57 | **0.55** | **0.575** |
| ROCKET | 0.58 | 0.54 | 0.568 |

The Shapelet-based model achieved the highest Macro F1-score and AUC, although all results remained relatively close to random classification.

This indicates that local movement subsequences contain some predictive information, but tabular variables provide a substantially stronger signal.

---

## Main Findings

- Tree-based ensemble models clearly outperformed linear classifiers.
- Gradient Boosting achieved the best multiclass classification performance.
- Sleep-related indicators were among the most important predictors.
- Resampling substantially improved minority-class recall in the imbalanced task.
- Accelerometer signals contain weak but detectable information about problematic Internet use.
- Shapelets were more informative than global time-series distances.
- Activity rhythms alone were not sufficient to produce well-separated clusters.
- The results describe statistical associations and should not be interpreted as clinical diagnoses.

---

## Suggested Repository Structure

```text
cmi-problematic-internet-use-analysis/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 00_data_understanding.ipynb
│   ├── 01_outlier_detection.ipynb
│   ├── 02_imbalanced_learning.ipynb
│   ├── 03_advanced_classification.ipynb
│   ├── 04_advanced_regression.ipynb
│   ├── 05_explainability.ipynb
│   ├── 06_time_series_preprocessing.ipynb
│   ├── 07_motifs_discords.ipynb
│   ├── 08_time_series_clustering.ipynb
│   └── 09_time_series_classification.ipynb
│
├── figures/
├── models/
├── report/
│   └── DM2Project_Tamberi.pdf
│
├── requirements.txt
├── .gitignore
└── README.md

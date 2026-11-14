# Project Report
## Machine Learning (CSE2032)

---

# TED Talks Smart Recommendation Engine Using Supervised Ensemble Re-Ranking and Natural Language Processing

### Department of Computer Science and Engineering
### School of Engineering and Technology
### BML Munjal University
**November 2026**

---

### Submitted By:
- **Student 1:** [Name of Student-1] | [Enrollment Number-1]
- **Student 2:** [Name of Student-2] | [Enrollment Number-2]
- **Student 3:** [Name of Student-3] | [Enrollment Number-3]
- **Student 4:** [Name of Student-4] | [Enrollment Number-4]

---

## Declaration by the Candidates

We hereby declare that the project entitled **"TED Talks Smart Recommendation Engine Using Supervised Ensemble Re-Ranking and Natural Language Processing"** has been carried out to fulfil the partial requirements for completion of the course **Machine Learning (CSE2032)** offered in the 5th Semester of the Bachelor of Technology (B.Tech) program in the Department of Computer Science and Engineering during Academic Year 2026-27 (odd semester). This experimental work has been carried out by us and submitted to the course instructor. Due acknowledgments have been made in the text of the project to all other materials used. This project has been prepared in full compliance with the requirements and constraints of the prescribed curriculum.

- **Candidate 1:** __________________________
- **Candidate 2:** __________________________
- **Candidate 3:** __________________________
- **Candidate 4:** __________________________

**Place:** BML Munjal University  
**Date:** _____ November, 2026

---

## Contents

| S.No. | Chapter / Section Title | Page No. |
| :---: | :--- | :---: |
| 1 | Introduction | 1 |
| 2 | Problem Statement & Objectives | 2-3 |
| 3 | Dataset Description & Attribute Analysis | 4 |
| 4 | Literature Review | 5-7 |
| 5 | Methodology & System Architecture | 8-16 |
| 6 | Technology Stack | 17 |
| 7 | Experimental Results & Comparative Evaluation | 18-25 |
| 8 | Team Responsibility / Self-Contribution Matrix | 26 |
| 9 | Conclusions & Future Work | 27 |
| - | References | 28 |
| - | Appendix: Source Code & Execution Flow | 29-32 |

---

## 1. Introduction

Online video and educational repositories have experienced exponential growth, leading to choice overload for learners and professionals. TED (Technology, Entertainment, Design) offers over 5,400 curated talks covering science, humanities, global issues, and emerging technologies. However, navigating this massive repository poses a significant discovery barrier.

Traditional recommender systems rely on explicit collaborative filtering (user ratings, watch histories, upvotes). In public video archives such as TED Talks, granular user-interaction logs are rarely public. A purely content-based keyword search often ignores engagement dynamics, recommending outdated or poorly presented talks.

This project implements a **Two-Stage Hybrid Machine Learning Recommendation System**:
1. **Stage 1 (Semantic Retrieval via NLP):** Uses Term Frequency-Inverse Document Frequency (TF-IDF) vectorization and Cosine Similarity to retrieve candidate talks strictly relevant to the user's topical query.
2. **Stage 2 (Supervised Popularity Re-Ranking):** Applies trained supervised machine learning classifiers—comparing Logistic Regression, Pruned Decision Trees, and Random Forest Ensembles—to evaluate and rank candidates based on engagement potential, presentation characteristics, and temporal attributes.

---

## 2. Problem Statement & Objectives

### 2.1 Problem Statement
Given a corpus of 5,429 TED Talks lacking individual user clickstream data, design an interactive machine learning pipeline that:
1. Accurately identifies conceptually relevant talks for any arbitrary topic or phrase.
2. Prioritizes talks that possess high audience reception and presentation quality without falling into the trap of target leakage.
3. Provides transparent comparative analysis across linear, rule-based, and ensemble machine learning paradigms.

### 2.2 Objectives
- **Data Engineering:** Clean and parse nested JSON fields (`topics`, `speakers`, `type`) from 5,430 raw records into structured tabular formats.
- **Leakage Prevention:** Establish a clean predictive target ($y = \text{is\_popular}$) based on the median split of `viewedCount` ($1,362,484$ views), strictly excluding view count and cosine similarity from the model training features.
- **Model Development & Comparison:** Build, tune, and evaluate three benchmark algorithms under 5-Fold Stratified Cross-Validation:
  - Logistic Regression (L2 Regularized Linear Classifier)
  - Decision Tree Classifier (Information Gain / Entropy with Pre-Pruning)
  - Random Forest Classifier (Bagging Ensemble of 150 Estimators)
- **Deployment:** Develop an interactive, production-ready web application using Streamlit featuring real-time inference, model consensus comparison, and session history management.

---

## 3. Dataset Description

The project utilizes the **Official TED Talks Dataset** sourced from Kaggle, containing video metadata across 27 initial attributes for 5,430 talks.

### 3.1 Attribute Inventory
- **Identifiers & Content:** `talk_id`, `socialTitle`, `description`, `topics` (JSON array of themes), `speakers` (JSON array of presenter info).
- **Temporal Features:** `publishedAt` (ISO 8601 release timestamp).
- **Format & Context:** `duration` (in seconds), `videoContext` (event edition, e.g., TED Global, TEDx, TED-Ed), `type` (talk format).
- **Engagement Ground Truth:** `viewedCount` (total views accumulated by the talk).

### 3.2 Target Definition
- **Continuous Distribution:** `viewedCount` is highly right-skewed, ranging from ~130,000 views to over 70 million views.
- **Target Discretization:** The median split threshold is established at $1,362,484$ views:
  $$\text{is\_popular} = \begin{cases} 1 & \text{if } \text{viewedCount} \ge 1,362,484 \\ 0 & \text{otherwise} \end{cases}$$
- This formulation establishes a perfectly balanced binary classification benchmark (50% Class 0, 50% Class 1), eliminating class imbalance bias.

---

## 4. Literature Review

From the reviewed studies, it can be seen that TED Talk recommendation has already been explored using different approaches. Some studies use collaborative filtering based on user ratings and preferences, while others use the content or transcripts of TED talks to find similar lectures. Hybrid methods have also been explored by combining content information with user preferences.

However, most of the existing TED recommendation approaches depend either on user interaction data or directly on similarity between the content of two talks. In our available dataset, public user ratings, clicks, and watch history are not available. This makes it difficult to directly apply traditional collaborative filtering methods or claim that the recommendations represent actual user preferences.

To address this situation, our project uses a different approach:
1. Instead of depending on unavailable user profiles, we use the available TED Talk content to retrieve relevant candidates using TF-IDF and Cosine Similarity.
2. We then engineer distinct structural features—such as duration, topic diversity, temporal publication trends, speaker count, and presentation context.
3. These features are supplied to supervised machine learning models to predict whether a talk belongs to the high-engagement tier.
4. **Target Leakage Mitigation:** The TF-IDF cosine similarity used to filter candidates is *never* provided to the machine learning models as an input feature. This allows the classifiers to learn patterns from presentation and structural attributes rather than simply memorizing keyword similarity.
5. Finally, instead of returning unranked similarity lists, the system uses predicted class probabilities to re-rank candidates and generate Top-N recommendations.

---

## 5. Methodology & System Architecture

```
[User Query] 
     │
     ▼
[Stage 1: NLP Retrieval] ──> TfidfVectorizer (5000 n-grams) ──> Cosine Similarity Matrix
                                                                       │
                                                            Top 30 Semantic Candidates
                                                                       │
                                                                       ▼
[Stage 2: Supervised Re-Ranking] <── Preprocessed Features (StandardScaler + OneHotEncoder)
     │
     ├── Logistic Regression (L2 Baseline)
     ├── Decision Tree (Entropy, Pruned)
     └── Random Forest Ensemble (150 Bagged Trees)
                                                                       │
                                                                       ▼
                                                          Ranked Top-N Recommendations + 
                                                          Consensus Matrix in Streamlit
```

### 5.1 Data Preprocessing & Parsing
- Extracted speaker names, topic counts, and event context types from JSON strings using Python's `ast.literal_eval`.
- Removed exact duplicates on `socialTitle` and `description` (yielding 5,429 clean talks).
- Calculated presentation duration in minutes: $\text{duration\_mins} = \frac{\text{duration}}{60.0}$.
- Derived temporal features from `publishedAt`: `published_year`, `published_month`, `published_dayofweek`, and `talk_age_days`.

### 5.2 Feature Engineering
The feature space fed to the supervised classifiers consists of:
- **Continuous Features:** `duration_mins`, `talk_age_days`, `published_year`, `published_month`, `published_dayofweek`, `topic_count`, `title_word_count`, `desc_word_count`.
- **Categorical Features:** `event_category` (top 8 contexts + 'Other'), `talk_type`.
- **Transformation Pipeline:**
  - Continuous variables are scaled using `StandardScaler` ($\mu=0, \sigma=1$).
  - Categorical variables are encoded using `OneHotEncoder(handle_unknown='ignore')`.

### 5.3 Supervised Classification Algorithms
1. **Logistic Regression (Baseline):**
   - Cost function regularized with $L_2$ penalty: $\min_{w} \frac{1}{2} w^T w + C \sum_{i=1}^n \log(1 + e^{-y_i w^T x_i})$.
   - Regularization parameter $C = 10.0$, solver `liblinear`.
2. **Decision Tree Classifier:**
   - Splitting criterion: Information Gain ($\text{Entropy} = -\sum p_i \log_2 p_i$).
   - Pruning controls: `max_depth=10`, `min_samples_leaf=50` to restrict branch growth and eliminate variance.
3. **Random Forest Classifier (Ensemble):**
   - $B = 150$ bootstrap trees with feature sub-sampling ($\text{max\_features} = \sqrt{d}$).
   - Prediction probability: $P(y=1|x) = \frac{1}{B} \sum_{b=1}^B P_b(y=1|x)$.

---

## 6. Technology Stack

- **Core Language:** Python 3.10+
- **Machine Learning & NLP:** `scikit-learn` (StandardScaler, OneHotEncoder, ColumnTransformer, Pipeline, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, TfidfVectorizer, cosine_similarity)
- **Data Manipulation & Analysis:** `pandas`, `numpy`
- **Frontend & Web Deployment:** `streamlit` (session state, layout grid, custom CSS badges)
- **Deployment Platform:** Streamlit Community Cloud (connected to GitHub CI/CD repository)

---

## 7. Experimental Results & Comparative Evaluation

All models were evaluated under **5-Fold Stratified Cross-Validation** to guarantee fair generalization across folds:

### 7.1 Cross-Validation Benchmark Summary

| Model | Mean Accuracy | Mean Precision | Mean Recall | Mean F1-Score | Mean ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (L2)** | 0.582 | 0.589 | 0.534 | 0.560 | 0.601 |
| **Decision Tree (Pruned)** | 0.658 | 0.667 | 0.631 | 0.648 | 0.722 |
| **Random Forest (Ensemble)** | **0.684** | **0.692** | **0.665** | **0.678** | **0.749** |

### 7.2 Key Findings
1. **Linear Separation Limitations:** Logistic Regression achieves an AUC of 0.601, indicating that audience popularity is non-linearly related to talk duration and publication dates.
2. **Tree Regularization:** Unpruned decision trees quickly overfit ($>95\%$ training accuracy but $<58\%$ test accuracy). Setting `min_samples_leaf=50` improved test generalization to 0.722 AUC.
3. **Ensemble Superiority:** Random Forest reduces variance through bagging, delivering the highest discrimination ability (AUC 0.749) and calibrated probability estimates.

---

## 8. Team Responsibility / Self-Contribution Matrix

| Project Activity | Member 1 | Member 2 | Member 3 | Member 4 |
| :--- | :---: | :---: | :---: | :---: |
| **Problem Definition & Formulation** | Primary | Contributor | Contributor | Contributor |
| **Data Ingestion, JSON Parsing & EDA** | Primary | Contributor | - | - |
| **Feature Engineering & Leakage Prevention** | Contributor | Primary | Contributor | - |
| **StandardScaler & OneHot Pipeline Setup** | - | Primary | Contributor | - |
| **Baseline Logistic Regression Modeling** | - | Contributor | Primary | - |
| **Decision Tree Tuning & Pruning** | - | - | Primary | Contributor |
| **Random Forest Ensemble Implementation** | - | - | Contributor | Primary |
| **TF-IDF & Cosine Retrieval Pipeline** | Contributor | - | - | Primary |
| **Streamlit Interface & Consensus Matrix** | - | Contributor | - | Primary |
| **Report Preparation & Technical Defense** | Contributor | Contributor | Primary | Primary |

---

## 9. Conclusions & Future Work

### 9.1 Conclusions
- We designed and deployed an end-to-end Machine Learning Recommendation Engine on 5,429 TED Talks without requiring private user clickstream logs.
- By decoupling semantic retrieval (Stage 1 TF-IDF) from supervised popularity estimation (Stage 2 Classifiers), we successfully eliminated target leakage.
- The Random Forest model achieved superior classification performance (AUC 0.749) compared to Decision Trees (0.722) and Logistic Regression (0.601).
- The deployed Streamlit web application allows users to query topics in real time, inspect model consensus, and manage their watch history.

### 9.2 Future Work
- Incorporating speech-to-text transcript analysis using transformer-based embeddings (e.g., MiniLM, Sentence-BERT).
- Exploring collaborative signals if anonymized watch sessions become publicly available.

---

## References
1. TED Conferences LLC. "TED Talks Complete Dataset", Kaggle Datasets, 2021.
2. Pedregosa, F., et al. "Scikit-learn: Machine Learning in Python", *Journal of Machine Learning Research*, 12:2825–2830, 2011.
3. Breiman, L. "Random Forests", *Machine Learning*, 45(1):5–32, 2001.
4. Manning, C. D., Raghavan, P., & Schütze, H. *Introduction to Information Retrieval*, Cambridge University Press, 2008.

---

## Appendix: Core Execution Flow

```python
# Stage 1: Semantic Candidate Retrieval
query_vec = tfidf.transform([user_query])
sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
candidate_indices = sim_scores.argsort()[::-1][:30]

# Stage 2: Supervised Popularity Re-Ranking
candidates_df = df_clean.iloc[candidate_indices].copy()
candidates_df['rf_prob'] = rf_pipe.predict_proba(X.iloc[candidate_indices])[:, 1]
top_recommendations = candidates_df.sort_values(
    by=['rf_prob', 'similarity'], 
    ascending=[False, False]
).head(top_n)
```

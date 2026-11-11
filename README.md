# 📩 SMS Spam Detection – Machine Learning

A machine learning project that classifies SMS messages as **Spam** or **Ham (Legitimate)** using Natural Language Processing (NLP) and multiple classification algorithms.

The project explores text preprocessing, feature extraction using TF-IDF, model training, and performance evaluation to identify the most effective machine learning classifier for SMS spam detection.

## 🎯 Project Overview

Spam messages are unwanted messages that may contain fraudulent offers, advertisements, or misleading information. This project aims to automatically distinguish spam messages from legitimate SMS messages using machine learning techniques.

The notebook implements an end-to-end classification workflow, starting from raw SMS data and progressing through text preprocessing, feature engineering, model training, and evaluation.

## ✨ Key Features

- **Data Cleaning:** Removes unnecessary columns and duplicate records.
- **Text Preprocessing:** Converts text to lowercase, tokenizes messages, removes punctuation and stopwords, and applies stemming.
- **Feature Extraction:** Uses TF-IDF Vectorization to convert text into numerical features.
- **Multiple ML Models:** Trains and compares 11 different classification algorithms.
- **Model Evaluation:** Evaluates models using accuracy and precision scores.

## 🛠️ Technologies & Libraries

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| NLTK | Natural Language Processing |
| Scikit-learn | Machine learning and TF-IDF |
| XGBoost | Gradient boosting classification |
| WordCloud | Text visualization |
| Jupyter Notebook | Development environment |

## 📂 Dataset

The project uses an SMS spam dataset (`spam.csv`) containing two primary columns:

- `v1`: Message classification (`ham` or `spam`).
- `v2`: Actual SMS message.

The dataset is cleaned and transformed into two main columns:

- `target`: Encoded classification labels.
- `text`: Original SMS messages.

### Data Preprocessing

The following preprocessing operations are performed:

1. Removal of unnecessary columns.
2. Encoding of target labels.
3. Identification and removal of duplicate records.
4. Conversion of text to lowercase.
5. Tokenization using NLTK.
6. Removal of special characters and stopwords.
7. Stemming using Porter Stemmer.

After removing duplicate records, the dataset contains **5,169 messages**.

## ⚙️ Machine Learning Workflow

### 1. Feature Extraction

TF-IDF (Term Frequency–Inverse Document Frequency) is used to transform processed SMS messages into numerical feature vectors.

Configuration:
- Maximum features: 500
- Vectorization technique: `TfidfVectorizer`

### 2. Train-Test Split

The dataset is divided into training and testing sets:

- Training data: 80%
- Testing data: 20%
- Random state: 2

### 3. Model Training

The following 11 classification algorithms are implemented and compared:

1. Support Vector Classifier (SVC)
2. K-Nearest Neighbors (KNN)
3. Multinomial Naive Bayes
4. Decision Tree
5. Logistic Regression
6. Random Forest
7. AdaBoost
8. Bagging Classifier
9. Extra Trees Classifier
10. Gradient Boosting Classifier
11. XGBoost

## 📊 Model Evaluation & Results

Models are evaluated using two performance metrics:

- **Accuracy:** Measures the percentage of correctly classified messages.
- **Precision:** Measures how many messages predicted as spam are actually spam.

### Performance Comparison

| Model | Accuracy | Precision |
|---|---:|---:|
| SVC | 96.62% | 93.28% |
| KNN | 92.75% | 100.00% |
| Naive Bayes | 97.10% | 96.55% |
| Decision Tree | 93.81% | 90.22% |
| Logistic Regression | 96.32% | 96.30% |
| Random Forest | 97.20% | 94.31% |
| AdaBoost | 96.13% | 93.75% |
| Bagging | 96.52% | 91.80% |
| Extra Trees | **97.29%** | 92.97% |
| Gradient Boosting | 95.07% | 93.94% |
| XGBoost | 97.10% | 95.76% |

### Key Observations

- Extra Trees achieved the highest accuracy of approximately **97.29%**.
- Multinomial Naive Bayes achieved approximately **97.10% accuracy** with 96.55% precision.
- KNN achieved 100% precision but had comparatively lower accuracy.
- Random Forest also demonstrated strong classification performance.

Based on accuracy, Extra Trees performed best among the evaluated models.

## 🚀 Getting Started

### Prerequisites

Make sure Python and Jupyter Notebook are installed on your system.

### 1. Clone the Repository

```bash
git clone https://github.com/SyedaFakhiraSaghir/SPAM-DETECTION-MLOPS.git
```

### 2. Navigate to the Project Directory

```bash
cd SPAM-DETECTION-MLOPS
```

### 3. Install

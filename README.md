# 📚 Topic Modeling of News Articles using Latent Dirichlet Allocation (LDA)

**Author:** Aditya Bhat  
**Dataset:** Harvard Dataverse – International News Articles  
**Objective:** Perform unsupervised topic modeling using Latent Dirichlet Allocation (LDA) on a corpus of news articles to uncover latent thematic structure and visualize topic trends over time.

---

## 🧠 Project Overview

This project applies **Latent Dirichlet Allocation (LDA)**, a probabilistic model for topic extraction, to a dataset of over 3,800 international news articles published during the politically significant period around the 2016 U.S. presidential transition.

Using Python and the Gensim library, the project follows a complete NLP pipeline:
- Data loading and preprocessing
- LDA model training
- Topic visualization (word clouds, trends)
- Evaluation using coherence and perplexity

---

## 🔧 Features

- **Tokenization, Stopword Removal, Lemmatization** using NLTK
- **Gensim LDA Model Training** with configurable topic count and passes
- **Interactive HTML Topic Visualization** using `pyLDAvis`
- **Word Clouds** for each topic
- **Temporal Topic Trends** using article timestamps
- **Model Evaluation** via coherence and perplexity

---

## 🗂️ Project Structure

```plaintext
lda_topic_modeling/
│
├── NewsArticles.csv                 # Input dataset (articles, dates, text)
├── lda_visualization.html          # Interactive topic visualization
├── topic_distribution_boxplot.png  # Boxplot of topic prevalence
├── topic_trend_over_time.png       # Time-series topic trend plot
├── wordcloud_topic_1.png           # Word cloud for Topic 1
├── ...
├── lda_pipeline.py                 # Main Python script (this repo)
└── README.md                       # Project overview and instructions

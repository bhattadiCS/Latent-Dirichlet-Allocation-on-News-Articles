"""
Author: Aditya Bhat
Project: Application of Latent Dirichlet Allocation (LDA) for Topic Modeling
Dataset: Harvard Dataverse - News Articles Dataset
Goal: Unsupervised topic extraction from news articles using probabilistic modeling (LDA)
"""

# ========== 1. Import Libraries ========== #

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

# ========== 2. Download NLTK Resources ========== #

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ========== 3. Helper Functions ========== #

def load_and_preprocess_data(file_path, text_column):
    """Load the dataset and preprocess text: tokenize, remove stopwords, lemmatize."""
    try:
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='utf-16')

    df = df.dropna(subset=[text_column])

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        tokens = word_tokenize(str(text).lower())
        return [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]

    df['tokens'] = df[text_column].apply(preprocess_text)
    return df

def train_lda_model(df_tokens, num_topics=5, passes=15):
    """Create dictionary and corpus, train LDA model."""
    dictionary = corpora.Dictionary(df_tokens)
    corpus = [dictionary.doc2bow(tokens) for tokens in df_tokens]

    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        alpha='auto',
        eta='auto'
    )

    return lda_model, corpus, dictionary

def visualize_and_save(lda_model, corpus, dictionary):
    """Create interactive topic visualization and save as HTML."""
    vis = gensimvis.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, 'lda_visualization.html')

def plot_topic_distribution(topic_df, num_topics):
    """Save boxplot showing topic distribution across documents."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=topic_df)
    plt.title('Topic Distribution Across Documents')
    plt.xlabel('Topic')
    plt.ylabel('Proportion')
    plt.xticks(ticks=range(num_topics), labels=[f"Topic {i+1}" for i in range(num_topics)])
    plt.tight_layout()
    plt.savefig("topic_distribution_boxplot.png")
    plt.close()

def generate_wordclouds(lda_model, num_topics):
    """Generate and save word cloud images for each topic."""
    for i in range(num_topics):
        plt.figure()
        wordcloud = WordCloud(background_color='white')
        plt.imshow(wordcloud.fit_words(dict(lda_model.show_topic(i, 50))))
        plt.axis('off')
        plt.title(f'Word Cloud - Topic {i + 1}')
        plt.tight_layout()
        plt.savefig(f"wordcloud_topic_{i + 1}.png")
        plt.close()

def plot_topic_time_series(df, topic_df, date_column):
    """Plot topic trends over time if date information exists."""
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df = df.dropna(subset=[date_column])
        df['dominant_topic'] = topic_df.idxmax(axis=1) + 1  # 1-indexed topics

        topic_trends = df.groupby([df[date_column].dt.to_period('M'), 'dominant_topic']).size().unstack(fill_value=0)

        topic_trends.plot(figsize=(12, 6))
        plt.title('Topic Prevalence Over Time (1-indexed)')
        plt.xlabel('Month')
        plt.ylabel('Number of Articles')
        plt.tight_layout()
        plt.savefig("topic_trend_over_time.png")
        plt.close()
    else:
        print(f"Column '{date_column}' not found. Skipping time series analysis.")

def evaluate_model(lda_model, corpus, df_tokens, dictionary):
    """Print model performance using perplexity and coherence score."""
    perplexity = lda_model.log_perplexity(corpus)
    coherence_model = CoherenceModel(model=lda_model, texts=df_tokens, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()

    print(f'\nModel Perplexity: {perplexity}')
    print(f'Model Coherence Score (c_v): {coherence_score}')

# ========== 4. Main Pipeline ========== #

def apply_lda(file_path='NewsArticles.csv', text_column='text', date_column='publish_date',
              num_topics=5, passes=15):
    """
    Full LDA Pipeline:
    1. Load and preprocess text data
    2. Train LDA model
    3. Visualize and interpret topics
    4. Evaluate model performance
    """

    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(file_path, text_column)

    print("Training LDA model...")
    lda_model, corpus, dictionary = train_lda_model(df['tokens'], num_topics, passes)

    print("\nDiscovered Topics:")
    for idx, topic in lda_model.print_topics(num_words=5):
        print(f"Topic {idx + 1}: {topic}")

    print("Saving interactive visualization...")
    visualize_and_save(lda_model, corpus, dictionary)

    topic_distributions = [lda_model.get_document_topics(bow) for bow in corpus]
    topic_df = pd.DataFrame([[dict(doc).get(i, 0) for i in range(num_topics)] for doc in topic_distributions])

    print("Generating plots...")
    plot_topic_distribution(topic_df, num_topics)
    generate_wordclouds(lda_model, num_topics)
    plot_topic_time_series(df, topic_df, date_column)

    print("Evaluating model...")
    evaluate_model(lda_model, corpus, df['tokens'], dictionary)

    print("\nAll outputs saved successfully.")

# ========== 5. Entry Point ========== #

if __name__ == "__main__":
    apply_lda(
        file_path='NewsArticles.csv',
        text_column='text',
        date_column='publish_date',
        num_topics=5,
        passes=15
    )

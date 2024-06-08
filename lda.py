import joblib
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.corpora import Dictionary
from gensim.models import LdaModel, TfidfModel
import pyLDAvis
import pyLDAvis.gensim_models
import pickle


# Load data
def load_data(filename):
    return pd.read_csv(filename)


# Text preprocessing function
def get_wordnet_pos(treebank_tag):
    """Converts treebank tags to wordnet tags."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # By default, NLTK's WordNetLemmatizer assumes everything is a noun.


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # regex to remove special characters
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    pos_tags = nltk.pos_tag(tokens)  # POS tagging
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]  # Lemmatization using POS tags
    tokens = [word for word in tokens if word not in stop_words]
    return tokens  # Return list of tokens


# Read data and preprocess texts
data = load_data('csv_files/train_v2_drcat_02.csv')
data['processed_text'] = data['text'].apply(preprocess_text)

# Create a dictionary and corpus for LDA
dictionary = Dictionary(data['processed_text'])
corpus = [dictionary.doc2bow(text) for text in data['processed_text']]

# Optionally, create a TF-IDF model and apply it to the corpus
tfidf = TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

# Set the number of topics
num_topics = 10

# Train the LDA model
lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

# If using TF-IDF
# lda_model = LdaModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary, passes=15)

# Extract the top words for each topic
topics = lda_model.show_topics(formatted=False)
topics_words = {i: [word for word, prob in topic] for i, topic in topics}

# Manually assign names to each topic based on the top words
topic_names = {
    0: "Technology and Services",
    1: "Electoral System",
    2: "Transportation and Pollution",
    3: "General Opinions",
    4: "Positive and Life Success",
    5: "Education and School",
    6: "Electoral College and Votes",
    7: "Cell Phone Usage",
    8: "Driverless Cars",
    9: "Space and Planets"
}

# Prepare LDA visualization data
vis_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)

# Save the LDA model and associated data
with open('pickle_file/lda_model.pkl', 'wb') as f:
    pickle.dump(lda_model, f)

with open('pickle_file/dictionary.pkl', 'wb') as f:
    pickle.dump(dictionary, f)

with open('pickle_file/corpus.pkl', 'wb') as f:
    pickle.dump(corpus, f)

with open('pickle_file/topic_names.pkl', 'wb') as f:
    pickle.dump(topic_names, f)

with open('pickle_file/lda_vis.pkl', 'wb') as f:
    pickle.dump(vis_data, f)

print("LDA model, dictionary, corpus, topic names, and visualization data have been saved.")

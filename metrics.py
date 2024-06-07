import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import gen_batches
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


# Download NLTK resources
def download_nltk_resources():
    resources = ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"]
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource)


download_nltk_resources()

# Read the CSV file
new_data = pd.read_csv('csv_files/train_v2_drcat_02.csv')


# Preprocess text function
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
    tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in
              pos_tags]  # Lemmatization using POS tags
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text


new_data['text'] = new_data['text'].apply(preprocess_text)

'''
print(new_data.head())

# Initialize and train the Naive Bayes classifier
nb_classifier = MultinomialNB()

# Determine the number of batches
n_samples = X_train.shape[0]
batch_size = 1000  # Adjust this value based on your memory capacity
n_batches = n_samples // batch_size

# Train the classifier in batches
for batch in gen_batches(n_samples, batch_size):
    nb_classifier.partial_fit(X_train[batch].toarray(), y_train[batch], np.unique(y_train))


tfidf_vectorizer = joblib.load('pickle_file/tfidf_vectorizer.pkl')
tfidf_vector = tfidf_vectorizer.transform([new_data['text']])
# Initialize and train the Logistic Regression classifier
logistic_regression_model = joblib.load('pickle_file/logistic_regression_model.pkl')

prediction1 = logistic_regression_model.predict(tfidf_vectorizer)
accuracy = accuracy_score(y_test, prediction1)
f1 = f1_score(y_test, prediction1)
recall = recall_score(y_test, prediction1)
precision = precision_score(y_test, prediction1)
print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
print(f'Recall: {recall}')
print(f'Precision: {precision}')

prediction = nb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, prediction)
f1 = f1_score(y_test, prediction)
recall = recall_score(y_test, prediction)
precision = precision_score(y_test, prediction)
print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
print(f'Recall: {recall}')
print(f'Precision: {precision}')'''

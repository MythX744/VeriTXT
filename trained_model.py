import joblib
import numpy as np
import pandas as pd
import re
import tensorflow as tf
import nltk
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import gen_batches
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


# Download necessary NLTK resources
def download_nltk_resources():
    resources = ["punkt", "stopwords", "wordnet"]
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource)


download_nltk_resources()


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
    tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in
              pos_tags]  # Lemmatization using POS tags
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text


# Read data and preprocess texts
data = load_data('csv_files/train_v2_drcat_02.csv')
data['text'] = data['text'].apply(preprocess_text)

# Vectorize texts using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(data['text'])
y = data['generated']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train Naive Bayes classifier in batches
def train_naive_bayes(X_train, y_train):
    naive_bayes_classifier = MultinomialNB()
    n_samples = X_train.shape[0]
    batch_size = 1000  # Adjust based on memory capacity
    for batch in gen_batches(n_samples, batch_size):
        naive_bayes_classifier.partial_fit(X_train[batch].toarray(), y_train[batch], np.unique(y_train))
    return naive_bayes_classifier


naive_bayes_model = train_naive_bayes(X_train, y_train)
joblib.dump(naive_bayes_model, 'pickle_file/naive_bayes_model.pkl')


# Train Logistic Regression classifier
def train_logistic_regression(X_train, y_train):
    logistic_regression_classifier = LogisticRegression()
    logistic_regression_classifier.fit(X_train, y_train)
    return logistic_regression_classifier


logistic_regression_model = train_logistic_regression(X_train, y_train)
joblib.dump(logistic_regression_model, 'pickle_file/logistic_regression_model.pkl')

# Save the TF-IDF vectorizer
joblib.dump(tfidf_vectorizer, 'pickle_file/tfidf_vectorizer.pkl')


# Train Doc2Vec model
def train_doc2vec_model(texts, vector_size=20, min_count=2, epochs=40):
    tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()), tags=[str(i)]) for i, doc in enumerate(texts)]
    model = Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epochs)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    return model


# First, ensure the Doc2Vec model is trained
doc2vec_model = train_doc2vec_model(data['text'])


# Function to generate document vectors using trained Doc2Vec model
def get_doc_vectors(model, texts):
    vectors = [model.infer_vector(word_tokenize(text.lower())) for text in texts]
    return vectors


# Function to generate training and test sets from Doc2Vec vectors
def prepare_training_data(doc_vectors, labels):
    # Convert list of vectors to numpy array for training
    X = np.array(doc_vectors)
    y = np.array(labels)
    return train_test_split(X, y, test_size=0.2, random_state=42)


# Generate document vectors for all data
all_doc_vectors = get_doc_vectors(doc2vec_model, data['text'])

# Prepare training and test data
X_train_NN, X_test_NN, y_train_NN, y_test_NN = prepare_training_data(all_doc_vectors, data['generated'])


# Train the Neural Network model
def train_neural_network(doc_vectors, labels):
    input_features = 20  # Assuming you're using 20-dimensional vectors from Doc2Vec
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(120, activation='relu', input_shape=(input_features,)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(76, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(doc_vectors, labels, epochs=3, batch_size=32)
    return model


# Train the neural network model with the correct data
neural_network_model = train_neural_network(X_train_NN, y_train_NN)

# Save the trained Doc2Vec and Neural Network models
joblib.dump(doc2vec_model, 'pickle_file/doc2vec_model.pkl')
joblib.dump(neural_network_model, 'pickle_file/neural_network_model.pkl')


# Metrics and evaluation
def evaluate_model(model, model_name, X_test, y_test, is_neural_net=False):
    prediction = model.predict(X_test)
    if is_neural_net:
        prediction = (prediction > 0.5).astype(int)  # Convert probabilities to binary labels

    accuracy = accuracy_score(y_test, prediction) * 100
    f1 = f1_score(y_test, prediction) * 100
    recall = recall_score(y_test, prediction) * 100
    precision = precision_score(y_test, prediction) * 100
    metrics = {
        'accuracy': f'{accuracy:.2f}%',
        'f1': f'{f1:.2f}%',
        'recall': f'{recall:.2f}%',
        'precision': f'{precision:.2f}%'
    }

    print(f'Accuracy: {accuracy:.2f}%')
    print(f'F1 Score: {f1:.2f}%')
    print(f'Recall: {recall:.2f}%')
    print(f'Precision: {precision:.2f}%')

    file_name = f'pickle_file/metrics_{model_name}.pkl'
    joblib.dump(metrics, file_name)


# Example usage
evaluate_model(naive_bayes_model, 'naive_bayes_model', X_test, y_test)
evaluate_model(logistic_regression_model, 'logistic_regression_model', X_test, y_test)
evaluate_model(neural_network_model, 'neural_network_model', X_test_NN, y_test_NN, is_neural_net=True)


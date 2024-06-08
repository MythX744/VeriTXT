import pickle
import sys
import pyLDAvis
import pyLDAvis.gensim
from flask import Flask, render_template, request
import joblib
import re
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Encoding configuration
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')


# Download NLTK resources
def download_nltk_resources():
    resources = ["punkt", "stopwords", "wordnet"]
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource)


download_nltk_resources()

# Load the models and vectorizer
naive_bayes_model = joblib.load('pickle_file/naive_bayes_model.pkl')
logistic_regression_model = joblib.load('pickle_file/logistic_regression_model.pkl')
tfidf_vectorizer = joblib.load('pickle_file/tfidf_vectorizer.pkl')
neural_network_model = joblib.load('pickle_file/neural_network_model.pkl')
doc2vec_model = joblib.load('pickle_file/doc2vec_model.pkl')
lr_metric = joblib.load('pickle_file/metrics_logistic_regression_model.pkl')
nb_metric = joblib.load('pickle_file/metrics_naive_bayes_model.pkl')
nn_metric = joblib.load('pickle_file/metrics_neural_network_model.pkl')

# Print model details
print("Loaded Naive Bayes Classifier:", naive_bayes_model)
print("Loaded Logistic Regression Model:", logistic_regression_model)
print("Logistic Regression Intercept:", logistic_regression_model.intercept_)
print("Logistic Regression Coefficients:", logistic_regression_model.coef_)
print("Loaded TF-IDF Vectorizer:", tfidf_vectorizer)
print("Loaded Neural Network Model:", neural_network_model)
print("Loaded Doc2Vec Model:", doc2vec_model)
print("Loaded Logistic Regression Metric:", lr_metric)
print("Loaded Naive Bayes Metric:", nb_metric)
print("Loaded Neural Network Metric:", nn_metric)


# Preprocess text function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text


# Predict NB function
def predictNB(text):
    preprocessed_text = preprocess_text(text)
    print(preprocessed_text)
    tfidf_vector = tfidf_vectorizer.transform([preprocessed_text])
    print(tfidf_vector)
    prediction = naive_bayes_model.predict(tfidf_vector.toarray())
    print(prediction)
    return "AI-generated" if prediction[0] > 0.5 else "Human-written"


# Predict LR function
def predictLR(text):
    preprocessed_text = preprocess_text(text)
    print(preprocessed_text)
    tfidf_vector = tfidf_vectorizer.transform([preprocessed_text])
    print(tfidf_vector)
    prediction = logistic_regression_model.predict(tfidf_vector)
    print(prediction)
    return "AI-generated" if prediction[0] > 0.5 else "Human-written"


# Predict NN function
def predictNN(text):
    try:
        preprocessed_text = preprocess_text(text)
        doc_vector = doc2vec_model.infer_vector(word_tokenize(preprocessed_text))
        X = np.array([doc_vector])  # Ensure this is 20-dimensional
        prediction = neural_network_model.predict(X)
        print(prediction)
        return "AI-generated" if prediction[0] > 0.5 else "Human-written"
    except Exception as e:
        error_message = f"Error during NN prediction: {str(e)}"
        try:
            print(error_message)  # Try printing the error message directly
        except UnicodeEncodeError:
            print(error_message.encode('utf-8', 'ignore').decode('utf-8', 'ignore'))  # Handle encoding issues
        raise Exception(error_message)  # Rethrow with the original message


# Load the LDA model and associated data
with open('pickle_file/lda_model.pkl', 'rb') as f:
    lda_model = pickle.load(f)

with open('pickle_file/dictionary.pkl', 'rb') as f:
    dictionary = pickle.load(f)

with open('pickle_file/corpus.pkl', 'rb') as f:
    corpus = pickle.load(f)

with open('pickle_file/topic_names.pkl', 'rb') as f:
    topic_names = pickle.load(f)

with open('pickle_file/lda_vis.pkl', 'rb') as f:
    vis_data = pickle.load(f)

# Extract the top words for each topic
topics = lda_model.show_topics(formatted=False)
topics_words = {i: [word for word, prob in topic] for i, topic in topics}


@app.route('/LDA', methods=['Post'])
def show_lda_vis():
    return render_template('lda_vis.html', vis_data=vis_data)

# Flask routes
@app.route('/')
def index():
    return render_template('new_home.html')


@app.route('/predict', methods=['POST'])
def predict_text():
    try:
        text = request.form['text']
        print(len(text))

        # Check if the text is empty
        if not text.strip():
            error_message = "Please enter some text."
            return render_template('new_home.html', error=error_message)

        # Limit the length of the text
        max_length = 500  # You can adjust the limit as needed
        min_length = 15
        if len(text) > max_length:
            error_message = f"Text is too long. Please limit your input to {max_length} characters."
            return render_template('new_home.html', error=error_message, input_text=text)

        if min_length > len(text):
            error_message = f"Text is too short. Please enter at least {min_length} characters."
            return render_template('new_home.html', error=error_message, input_text=text)

        model = request.form['model']
        prediction = "Model not recognized"
        if model == 'Naive Bayes':
            prediction = predictNB(text)
        elif model == 'Logistic Regression':
            prediction = predictLR(text)
        elif model == 'Neural Networks':
            prediction = predictNN(text)

        return render_template('new_home.html', prediction=prediction, input_text=text)
    except Exception as e:
        print(e)
        return str(e), 500


@app.route('/EDA', methods=['POST'])
def EDA():
    try:
        model = request.form['model']
        if model == 'EDA':
            return render_template('new_eda.html')
    except Exception as e:
        print(e)
        return str(e), 500


@app.route('/Metrics', methods=['POST'])
def metrics():
    try:
        model = request.form['model']
        if model == 'Metrics':
            # Logistic Regression
            accuracy_lr = lr_metric['accuracy']
            f1_lr = lr_metric['f1']
            recall_lr = lr_metric['recall']
            precision_lr = lr_metric['precision']
            print(f"Accuracy : {accuracy_lr}")
            print(f"F1 Score : {f1_lr}")
            print(f"Recall : {recall_lr}")
            print(f"Precision : {precision_lr}")
            # Naive Bayes
            accuracy_nb = nb_metric['accuracy']
            f1_nb = nb_metric['f1']
            recall_nb = nb_metric['recall']
            precision_nb = nb_metric['precision']
            print(f"Accuracy: {accuracy_nb}")
            print(f"F1 Score: {f1_nb}")
            print(f"Recall: {recall_nb}")
            print(f"Precision: {precision_nb}")
            # Neural Network
            accuracy_nn = nn_metric['accuracy']
            f1_nn = nn_metric['f1']
            recall_nn = nn_metric['recall']
            precision_nn = nn_metric['precision']
            print(f"Accuracy: {accuracy_nn}")
            print(f"F1 Score: {f1_nn}")
            print(f"Recall: {recall_nn}")
            print(f"Precision: {precision_nn}")
            return render_template('new_metrics.html', accuracy_lr=accuracy_lr, f1_lr=f1_lr, recall_lr=recall_lr,
                                   precision_lr=precision_lr, accuracy_nb=accuracy_nb, f1_nb=f1_nb, recall_nb=recall_nb,
                                   precision_nb=precision_nb, accuracy_nn=accuracy_nn, f1_nn=f1_nn, recall_nn=recall_nn,
                                   precision_nn=precision_nn)
    except Exception as e:
        print(e)
        return str(e), 500



if __name__ == '__main__':
    app.run(debug=True)

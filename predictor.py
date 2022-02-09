import subprocess

import spacy

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

class Predictor:
    '''
    This class is used to train and use a predictor model to predict topics in sustainability reports.
    '''
    def __init__(self, labels, vectorization_mode = 'bag_of_words', prediction_mode = 'logistic_regression', optimizer_mode = 'none', spacy_data = 'en_core_web_md', max_features = 1000):
        '''
        Parameters:
            labels: Specifies what labels the model should be trained on. These labels need to be present in the training data set
            vectorization_mode: A string specifying the vectorization mode of the text. Default: bag_of_words. Alternative: 'spacy_document_vectors'
            prediction_mode:  A String specifying the prediction approach. Default: logistic regression. Alternative: naive_bayes
            optimizer_mode: A String specifying a performance metric to optimize the model for. Default: none. Alternatives: accuracy, recall
            spacy_data: A string specifying the spacy language model to be used. Default: en_core_web_md. Alternatives: https://spacy.io/usage/models
            max_features: The maximum numbe of vector features if bag of words vectorization is used. Default: 1000
        '''
        # load spacy language model if not already in cache
        try:
            self.nlp = spacy.load(spacy_data)
        except:
            subprocess.call(['python', '-m', 'spacy', 'download', spacy_data])
            self.nlp = spacy.load('en_core_web_md')
            pass

        self.labels = labels # labels for which predictions can be generated
        self.vectorization_mode = vectorization_mode
        self.prediction_mode = prediction_mode
        self.models = []
        self.thresholds = []
        self.count_vectorizer = CountVectorizer(max_features = max_features)
        self.optimizer_mode = optimizer_mode


    def tokenize(self, texts):
        '''
        This function tokenizes a list of texts and returns for each text a string of tokens, separated by whitespaces.

        Tokenization_steps: to lower string > remove stop_words > lammatization > stripping
        '''
        tokens = []
        texts = [text.lower() for text in texts]
        for doc in self.nlp.pipe(texts):
            token_vector = [word.lemma_.strip() for word in doc if word.is_punct == False and word.is_stop == False and not word.lemma_.isnumeric()]
            tokens.append(' '.join(token_vector))

        return tokens


    def bow_vectorizer_fit_transform(self, tokens):
        '''
        This function fits the classes bag of words model and vectorizes a list of strings
        '''
        vectors = self.count_vectorizer.fit_transform(tokens).toarray()
        return vectors


    def bow_vectorizer_transform(self, tokens):
        ''' 
        This function vectorizes a list of strings using the classes bag of words model
        '''
        vectors = self.count_vectorizer.transform(tokens).toarray()
        return vectors

    def doc_vectorizer_transform(self, tokens):
        '''
        This function uses the spacy language model to vectorize a list of strings
        '''
        vectors = []

        for doc in self.nlp.pipe(tokens):
            vector = doc.vector
            vectors.append(vector)

        return vectors

    def optimize_threshold(self, clf, vectors, training_labels):
        '''
        Both available prediction models of this class generate probabilities for the likelihood of being labeled.
        This function optimizes the probability threshold for labeling. The optimization metric is set during the
        instantiation of the class.
        '''
        top_perf = 0
        top_threshold = 0

        # if optimization mode is set to none, optimize nothing
        if self.optimizer_mode != 'none':

            # Optimize for probability thresshold 0 to 1 in 0.01 steps
            for threshold in range(0, 100, 1):

                thresh = threshold/100
                
                # Predict labels based on temporary threshold
                pred = clf.predict_proba(vectors)
                pred = [1 if proba[1] > thresh else 0 for proba in pred]

                # Calculate performance metrics based on optimization mode
                if self.optimizer_mode == 'accuracy':
                    perf = sum([1 if training_labels[i] == pred[i] else 0 for i in range(len(training_labels))])/len(training_labels)

                elif self.optimizer_mode == 'recall':
                    perf = sum([1 if (training_labels[i] == pred[i] and training_labels[i] == 1) else 0 for i in range(len(training_labels))])/sum(training_labels)

                # Keep best performing threshold
                if perf >= top_perf:
                    top_perf = perf
                    top_threshold = thresh

        return top_threshold

    def train(self, texts, text_labels):
        '''
        This function trains the labeling model. The hyperparameters of the model are set during the instantiation of the class instance

        Parameters:
            texts: A list of texts which are vectorized
            text_labels: A list of labels which are used for training

        '''
        # Clenase and tokenize text
        tokenized_texts = self.tokenize(texts)

        # Choose a vectorization method depending on the hyperparameters
        if self.vectorization_mode == 'bag_of_words':
            vectors = self.bow_vectorizer_fit_transform(tokenized_texts)
        elif self.vectorization_mode == 'spacy_document_vectors':
            vectors = self.doc_vectorizer_transform(tokenized_texts)

        # Train a model for each label
        for label in self.labels:

            # Turn labels into numeric categories
            training_labels = []
            for i in range(len(vectors)):
                if text_labels[i] == label:
                    training_labels.append(1)
                else:
                    training_labels.append(0)

            # Choose and fit a model depending on the hyperparameters
            if self.prediction_mode == 'logistic_regression':
                clf = LogisticRegression(class_weight=None)
                clf.fit(vectors, training_labels)
            elif self.prediction_mode == 'naive_bayes':
                clf = GaussianNB()
                clf.fit(vectors, training_labels)
            
            self.models.append(clf)
            self.thresholds.append(self.optimize_threshold(clf, vectors, training_labels))


    def predict(self, texts, label, proba = False, label_names=True):
        '''
        This method predicts labels for a list of texts and returns them as a list

         Parameters:
            texts:  A list of texts to label
            label: The label which should be predicted. The label must be one of the labels for which the model is trained for
            proba: if True, the probabilities of the label are returned. If False, the label is returned. Defualt: False
            label_names: if True, the name of the label is used in the returned list. If false, numeric classes are returned. Default: True

        '''

        # Cleanse and tokenize texts
        tokenized_texts = self.tokenize(texts)
        
        # Vectorize texts depending on the chosen hyperparameters
        if self.vectorization_mode == 'bag_of_words':
            vectors = self.bow_vectorizer_transform(tokenized_texts)
        elif self.vectorization_mode == 'spacy_document_vectors':
            vectors = self.doc_vectorizer_transform(tokenized_texts)
        
        # Select model and thresshold for the label to predict
        model = self.models[self.labels.index(label)]
        threshold = self.thresholds[self.labels.index(label)]

        # Return probabilities if proba is set to True. Otherwise, return labels
        if proba:
            predictions = model.predict_proba(vectors)
        else:
            # If no optimization is used, use no thresholds. Otherwise, use thresholds
            if self.optimizer_mode == 'none':
                predictions = model.predict(vectors)
            else:
                predictions = model.predict_proba(vectors)
                predictions = [1 if proba[1] > threshold else 0 for proba in predictions]
            
            # If label_names is True, return label instead of numberic categories
            if label_names:
                predictions = [label if p == 1 else None for p in predictions]

        return predictions



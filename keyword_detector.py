'''
This class is used to categorize texts based on keywords fount through Wikipedia articles.

It was initially created for the businessresponsibility.ch project of the Prototype Fund. For more
information visit https://github.com/bizres

'''

import subprocess
import warnings

import spacy
import wikipediaapi
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

class KeywordDetector:
    '''
    This class is used to categorize report paragraphs, based on keywords extracted from Wikipedia
    '''
    def __init__(self, lang = 'en'):
        '''
        On instantiation the class loads the required language model from spacy. If the spacy model is not downloaded, the class
        will try to download it in a subprocess. If this fails, the spacy model has to be downloaded by the user: https://spacy.io/usage/models#download

        Parameters:
            lang: The language model used to cleanse text. Possible values are en, de, fr and it. 

        '''
        # Suspported languages
        self.supported_languages = {'en': 'en_core_web_md', 'de': 'de_core_web_md', 'fr': 'fr_core_web_md', 'it': 'it_core_web_md'}
        # Language specified during instantiation
        self.lang = lang
        # Topics loaded during topic loading
        self.topics = []
        # Base texts for topic extraction
        self.topic_texts = []
        # Extracted keywords for topic
        self.topic_keywords = []
        
        if self.lang in self.supported_languages.keys():
            # load spacy language model
            try:
                # load spacy language model
                self.nlp = spacy.load(self.supported_languages[self.lang])
            except:
                # download spacy language model if not downloaded yet
                subprocess.call(['python', '-m', 'spacy', 'download', self.supported_languages[self.lang]])
                self.nlp = spacy.load('en_core_web_md')

            self.wiki_wiki = wikipediaapi.Wikipedia(self.lang)

        else:
            raise ValueError('The specified language is not supported by this class. Supported languages are: {}'.format(self.supported_languages.keys()))


    def get_topics(self):
        '''
        Get topics which the class can detect
        '''
        return self.topics


    def get_topic_texts(self):
        '''
        Get texts used for topic modelling
        '''
        return self.topic_texts


    def get_topic_keywords(self):
        '''
        Get keywords which represent the topics
        '''
        return self.topic_keywords


    def __cleanse_text(self, text, min_char_ = 100):
        '''
            This method cleansed text using a spacy language model. The process is: tokenization > remove numbers > remove punctuation > remove stop words > > lemmatization > to lower

            Parameters:
            min_char_: The minimum amount of chars a text should have. Less characters result in a returned empty string
        
        '''
        if len(text) < min_char_:
            return ''

        # Parse text with spacy
        doc = self.nlp(text, disable=['parser', 'ner'])

        # Iterate over all words in text
        tokens = []
        for token in doc:
            # Remove stop words, punctuation and numbers
            if (not token.is_stop) and (not token.is_punct) and (not token.text.replace('.','',1).isdigit()):
                # lemmatize
                tokens.append(token.lemma_.lower())

        # Return as string
        return ' '.join(tokens)


    def __sort_coo(self, coo_matrix):
        '''
        This sort keyword tuples by their values

        Attribution: kavita-ganesan.com/python-keyword-extraction
        '''
        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
        
    def __extract_from_vector(self, feature_names, sorted_items, min_tf_idf_):
        '''
        Extract feature names from keyword vector.

        Attribution: kavita-ganesan.com/python-keyword-extraction

        Parameters:
        min_tf_idf_: Minimum tf_idf value a keyword must have. Lesser values are ignored
        '''
        
        #use only topn items from vector
        sorted_items = sorted_items
        score_vals = []
        feature_vals = []
        
        # word index and corresponding tf-idf score
        for idx, score in sorted_items:
            #keep track of feature name and its corresponding score
            if score >= min_tf_idf_:
                score_vals.append(round(score, 3))
                feature_vals.append(feature_names[idx])
        #create a tuples of feature,score
        #results = zip(feature_vals,score_vals)
        results= {}
        for idx in range(len(feature_vals)):
            results[feature_vals[idx]]=score_vals[idx]
        
        return results

    def __calculate_keywords(self, texts, max_df_=0.7, ngram_range_=(2, 2), min_tf_idf_=0.08, max_keywords_=10):
        '''
        This method extracts keywords from provided texts using tf-idf score. For more information see https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

        Parameters:
        texts: list of texts to process
        max_df_: Words which are present in max_df_ percent of texts (i.e. 0.7) are ignored
        ngram_range: range of ngrams to consider (tuple)
        min_tf_idf_: Minimum tf_idf score which keywords must have
        max_keywords_: Maximum number of keywords which are returned
        '''

        # Instantiate vectorizer and tf-idf-counter
        cv = CountVectorizer(max_df=max_df_, ngram_range=ngram_range_)
        tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)

        # Build bow for whole corpus of texts
        word_count_vector = cv.fit_transform(texts)

        # Build tf-idf scores
        tfidf_transformer.fit(word_count_vector)

        # get names of tf-idf features
        feature_names=cv.get_feature_names_out()

        keywords = []

        # get keywords for each text
        for text in texts:
            # Get bow of document
            tf_idf_vector=tfidf_transformer.transform(cv.transform([text]))
            # Sort features by tf-idf score
            sorted_items=self.__sort_coo(tf_idf_vector.tocoo())
            # Get feature names from vector and filter them by minimal tf-idf score
            words = self.__extract_from_vector(feature_names,sorted_items, min_tf_idf_)

            keywords.append(words)
        
        # Only use top max_keywords_ keywords
        keywords_filtered = []
        for words in keywords:
            filt = sorted(words.items(), key=lambda x: x[1], reverse=True)[:max_keywords_]
            filt = [word[0] for word in filt]
            keywords_filtered.append(filt)

        return keywords_filtered    


    def load_topics(self, topics_, topic_articles_, min_words = 1000, min_tf_idf_ = 0.08, max_keywords_ = 10, min_keywords_ = 2, ngram_range_ = (2, 2), max_df_=0.7):
        '''
        This method loads topics using wikipedia articles. The class remembers the topics and keywords (using tf-idf score) to detect topics with the detect_keywords function
        
        Parameters:
        topics_: List with name of topics
        topic_articles: List with Wikipedia articles used for topic modelling
        min_words: Minimum number of words a Wikipedia article must have to be considered
        min_tf_idf_: Miminum tf-idf score a keyword must have to be considered
        max_keywords_: The maximum number of generated keywords by topic
        min_keywords_: The minimum number of keywords a topic must have to be considered
        ngram_range_: The ngram range for more information see https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
        max_df: Filter option to ignore common words (i.e. 0.7 means no words which are present in more than 70% of the texts)
        '''

        # Raise error if number of topics does not match the number of provided articles
        if len(topics_) != len(topic_articles_):
            raise ValueError('Number of topics must ber the same as number of articles. Number of topics: {}, Number of articles: {}'.format(len(topics_), len(topic_articles_)))
        
        topics = []
        topic_texts = []

        # Download text of wikipedia articles
        for i in range(len(topics_)):
            wiki_page = self.wiki_wiki.page(topic_articles_[i])
            text = wiki_page.text
            text = self.__cleanse_text(text, 0)

            # Ignore text if minimum number of words is not reached
            if len(text.split(' ')) >= min_words:
                topics.append(topics_[i])
                topic_texts.append(text)
            else:
                warnings.warn('Topic \"{}\" was excluded due to not enough text (necessary words: {}, actual words: {}'.format(topics_[i], min_words, len(text.split(' '))))

        # Generate Keywords
        topic_keywords = self.__calculate_keywords(topic_texts, max_df_=max_df_, ngram_range_=ngram_range_, min_tf_idf_=min_tf_idf_, max_keywords_=max_keywords_)
        
        # Ignore topics if less than min_keywords_ keywords were generated
        for i in range(len(topic_keywords)):
            if len(topic_keywords[i]) >= min_keywords_:
                self.topics.append(topics[i])
                self.topic_texts.append(topic_texts[i])
                self.topic_keywords.append(topic_keywords[i])
            else:
                warnings.warn('Topic \"{}\" was excluded since less than {} keywords could be extracted.'.format(topics[i], min_keywords_))


    def detect_keywords(self, df, column, min_char_):
        '''
        This method expects a pandas dataframe with multiple texts. The method returns a pandas dataframe
        with additional columns for each loaded topic. The columns contain the number of unique keywords
        found in the texts.

        Parameters:
        df: The pandas dataframe to process
        column: The name of the column which contains the texts
        min_char_: Texts with less charactes than min_char_ will not be processed
        '''

        # Cleanse and tokenize the texts
        df['cleansed_text'] = df[column].apply(self.__cleanse_text, min_char_ = min_char_)

        # Search for keywords and add a column for each topic
        for i in range(len(self.topics)):
            # We start with a vector of zeroes
            matches = np.zeros(len(df))
            for keyword in self.topic_keywords[i]:
                # Test if texts contain keyword
                keyword_matches = df['cleansed_text'].str.contains(keyword).to_numpy()
                # Count number of found unique keywords
                matches = matches + keyword_matches
            # Add vector as column to pandas dataframe
            df[self.topics[i]] = matches
        
        return df
        
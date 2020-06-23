# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 17:28:12 2020

@author: Sidhhant Bhatnagar

"""
# Dataset used:- essays_&_scores.csv

# For this project, only essay set 1 (out of 8) was used for analysis and model creation.

# Features:
# 1. Bag of Words (BOW) counts (10000 words)
# 2. Number of characters in an essay
# 3. Number of words in an essay
# 4. Number of sentences in an essay
# 5. Number of lemmas in an essay
# 6. Number of spellng errors in an essay
# 7. Number of nouns in an essay
# 8. Number of adjectives in an essay
# 9. Number of verbs in an essay
# 10.Number of adverbs in an essay

'''
We used Lemmatization instead of Stemming as it is:
pros:-    MORE ACCURATE
         ALWAYS REDUCES TO DICTIONARY WORD
        IT DOESN'T JUST OMIT THE END LETTERS OF A WORD LIKE STEMMING
    
cons:- VERY TIME CONSUMING
''''

import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re, collections
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import cohen_kappa_score
from sklearn.svm import SVR
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV

dataframe = pd.read_csv('C:/Users/sidhh/Desktop/FORSK PROJECT/essays_&_scores.csv', encoding = 'latin-1')

# extracting useful data only

data = dataframe[['essay_set','essay','domain1_score']].copy()

# Tokenize a sentence into words

def sentence_to_wordlist(raw_sentence):
    
    clean_sentence = re.sub("[^a-zA-Z0-9]"," ", raw_sentence)
    tokens = nltk.word_tokenize(clean_sentence)
    
    return tokens

# tokenizing an essay into a list of word lists

def tokenize(essay):
    stripped_essay = essay.strip()
    
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(stripped_essay)
    
    tokenized_sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            tokenized_sentences.append(sentence_to_wordlist(raw_sentence))
    
    return tokenized_sentences

# calculating number of words in an essay

def word_count(essay):
    
    clean_essay = re.sub(r'\W', ' ', essay)
    words = nltk.word_tokenize(clean_essay)
    
    return len(words)

# calculating number of characters in an essay

def char_count(essay):
    
    clean_essay = re.sub(r'\s', '', str(essay).lower())
    
    return len(clean_essay)


# calculating number of sentences in an essay

def sent_count(essay):
    
    sentences = nltk.sent_tokenize(essay)
    
    return len(sentences)


# calculating number of lemmas per essay

def count_lemmas(essay):
    
    tokenized_sentences = tokenize(essay)      
    
    lemmas = []
    wordnet_lemmatizer = WordNetLemmatizer()
    
    for sentence in tokenized_sentences:
        tagged_tokens = nltk.pos_tag(sentence) 
        
        for token_tuple in tagged_tokens:
        
            pos_tag = token_tuple[1]
        
            if pos_tag.startswith('N'): 
                pos = wordnet.NOUN
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('J'):
                pos = wordnet.ADJ
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('V'):
                pos = wordnet.VERB
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('R'):
                pos = wordnet.ADV
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            else:
                pos = wordnet.NOUN
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
    
    lemma_count = len(set(lemmas))
    
    return lemma_count


# checking number of misspelled words

def count_spell_error(essay):
    
    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)
    
    #big.txt: It is a concatenation of public domain book excerpts from Project Gutenberg 
    #         and lists of most frequent words from Wiktionary and the British National Corpus.
    #         It contains about a million words.
    data = open('big.txt').read()
    
    words_ = re.findall('[a-z]+', data.lower())
    
    word_dict = collections.defaultdict(lambda: 0)
                       
    for word in words_:
        word_dict[word] += 1
                       
    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)
                        
    mispell_count = 0
    
    words = clean_essay.split()
                        
    for word in words:
        if not word in word_dict:
            mispell_count += 1
    
    return mispell_count


# calculating number of nouns, adjectives, verbs and adverbs in an essay

def count_pos(essay):
    
    tokenized_sentences = tokenize(essay)
    
    noun_count = 0
    adj_count = 0
    verb_count = 0
    adv_count = 0
    
    for sentence in tokenized_sentences:
        tagged_tokens = nltk.pos_tag(sentence)
        
        for token_tuple in tagged_tokens:
            pos_tag = token_tuple[1]
        
            if pos_tag.startswith('N'): 
                noun_count += 1
            elif pos_tag.startswith('J'):
                adj_count += 1
            elif pos_tag.startswith('V'):
                verb_count += 1
            elif pos_tag.startswith('R'):
                adv_count += 1
            
    return noun_count, adj_count, verb_count, adv_count
    


# getiing Bag of Words (BOW) counts

def get_count_vectors(essays):
    
    vectorizer = CountVectorizer(max_features = 10000, ngram_range=(1, 3), stop_words='english')
    # ngram_range = unigram,bigram,trigram
    
    count_vectors = vectorizer.fit_transform(essays) #transforms into matrix
    
    feature_names = vectorizer.get_feature_names() #all unique words
    
    return feature_names, count_vectors


# splitting Bag Of Words data into train data and test data 

feature_names_cv, count_vectors = get_count_vectors(data[data['essay_set'] == 1]['essay'])
X_cv = count_vectors.toarray()
y_cv = data[data['essay_set'] == 1]['domain1_score']
X_train, X_test, y_train, y_test = train_test_split(X_cv, y_cv, test_size = 0.3)


# Training linear regression model using ONLY BOW

linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
y_pred = linear_regressor.predict(X_test)


print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print('Cohen\'s kappa score: %.2f' % cohen_kappa_score(np.rint(y_pred), y_test))
print('R2 score: %.2f' % r2_score(y_test,y_pred))v

# ------- POOR RESULTS.--------------


# extracting essay features

def extract_features(data):
    features = data.copy()
     
     
    features['char_count'] = features['essay'].apply(char_count)
    
    features['word_count'] = features['essay'].apply(word_count)
    
    features['sent_count'] = features['essay'].apply(sent_count)
    
    features['lemma_count'] = features['essay'].apply(count_lemmas)
    
    features['spell_err_count'] = features['essay'].apply(count_spell_error)
    
    features['noun_count'], features['adj_count'], features['verb_count'], features['adv_count'] = zip(*features['essay'].map(count_pos))
    
    return features

#taking only features of essay_set 1 
feautures_set1= extract_features(data[data['essay_set']==1])

#took 20 mins to extract features.

feautures_set1.plot.scatter(x = 'char_count', y = 'domain1_score', s=10)
feautures_set1.plot.scatter(x = 'word_count', y = 'domain1_score', s=10)
feautures_set1.plot.scatter(x = 'sent_count', y = 'domain1_score', s=10)
feautures_set1.plot.scatter(x = 'lemma_count', y = 'domain1_score', s=10)
feautures_set1.plot.scatter(x = 'spell_err_count', y = 'domain1_score', s=10)
feautures_set1.plot.scatter(x = 'noun_count', y = 'domain1_score', s=10)
feautures_set1.plot.scatter(x = 'adj_count', y = 'domain1_score', s=10)
feautures_set1.plot.scatter(x = 'verb_count', y = 'domain1_score', s=10)
feautures_set1.plot.scatter(x = 'adv_count', y = 'domain1_score', s=10)

#By plotting the above scatter plots, we can examine how different features above affect the grade the student receives.

# 1. I observed that there is a strong correlation between character count of an essay and the final essay score. 
#    I observed similar correlations for word count, sentence count and lemma count of an essay.
#    These features indicate language fluency.
# 2. Various parts-of-speech such as nouns, adjectives adverbs and verbs are good features to test vocabulary
#    these parameters have strong co-relation with final essay score.
# 4. There is a weaker correlation between the number of spelling errors and the final score of an essay
#     which is surprising !!      



# NOW we will split ALL FEATURES (Bag Of Words + Other features) into train and test data

X = np.concatenate((feautures_set1.iloc[:, 3:], X_cv), axis = 1)
y = feautures_set1['domain1_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


# Training a Linear Regression model using all the features (BOW + other features)

linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
y_pred = linear_regressor.predict(X_test)


print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print('Cohen\'s kappa score: %.2f' % cohen_kappa_score(np.rint(y_pred), y_test))
print('R2 score:  %.2f ' % r2_score(y_test,y_pred))

#--- RESULTS BETTER THAN BEFORE  ---- 


# training a SVR model using all features (BOW + other features)

# ------ took 35 minutes to train this model.

svr = SVR()

parameters = {'kernel':['linear', 'rbf'], 'C':[1, 100], 'gamma':[0.1, 0.001]}

grid = GridSearchCV(svr, parameters)
grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)

# summarize the results of the grid search
print(grid.best_score_)
print(grid.best_estimator_)

print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print('Cohen\'s kappa score: %.2f' % cohen_kappa_score(np.rint(y_pred), y_test))
print('R2 score:  %.2f ' % r2_score(y_test,y_pred))

#--- better score than linear regression----


# Observations:
    
# 1. Using only BOW features:
#     linear regression performs poorly.

# 2. Using all features:
#     Linear Regression's scores are better than before
#     SVR performs better (higher score), but VERY time consuming







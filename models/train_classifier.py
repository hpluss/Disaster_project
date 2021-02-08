import sys
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV



def load_data(database_filepath):
    '''
    Loading data from the Database
    INPUT :
        database_filepath [string]: The path of the database
    
    OUTPUT:
        X,Y,Y.columns : A tuple that contains a list of messages (X), dataframe of categories by message (Y),
                        a list a categories (Y.columns)

    '''
    # load data at database_filepath
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('messages', engine)
    X = df.message
    Y = df.iloc[:,4:]
    
    return X, Y, Y.columns


def tokenize(text):
    '''
    Return a list of word after applying tokenize techniques to the input text
    INPUT :
        text [string]: A user message
    
    OUTPUT:
        tokens [list] : A list of word

    '''

    # Normalize
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # Tokenization
    words = word_tokenize(text)
    
    # Stop word removal
    words = [w for w in words if w not in stopwords.words("english")]
    
    # Lemmatization
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    tokens = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]
    
    # Stemming
    # tokens = [PorterStemmer().stem(w) for w in lemmed]
    
    return tokens


def build_model():
    '''
    Returns a pipeline containing the different steps of the model 
    INPUT:
    
    OUTPUT:
        model [Pipeline] 

    '''

    model =  Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])
    
    #parameters = param_grid = { 
    #'clf__estimator__max_features': ['auto','log2']
    #}

    #model = GridSearchCV(model, param_grid=parameters)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Display accuracy, F1-score and the best parameters of the model
    INPUT:
        model : the model to analyze
        X_test : data to be tested
        Y_test: True result for the test data
        category_names [list] : a list of all categories

    OUTPUT:
       

    '''

    test_pred = model.predict(X_test) 

    for i in range(len(category_names)): 
        print(category_names[i]) 
        print(classification_report(Y_test[category_names[i]], test_pred[:, i]))
    
    return


def save_model(model, model_filepath):
    '''
    Save the model to indicated path
    INPUT:
        model : the model to save
        model_filepath [string]: The location where to save the model

    OUTPUT:

    '''
    # save model to model_filepath
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

    return


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
import sys
import pandas as pd
import numpy as np
import pickle
import re
import nltk
from sqlalchemy import create_engine
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

nltk.download('stopwords')

def load_data(database_filepath):
    """
        Load data from the sqlite database. 
    Input: 
        database_filepath: the path of the database file
    Returns: 
        X: messages 
        Y: categories
        category_names (List)
    """
    
    # load data from database
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql_table('DisasterResponse', engine)
    df = df[~(df == 2).any(axis=1)]
    df = df.drop(columns=['child_alone'])
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    Normalizes, tokenizes and lemms text
    
    Input:
        text: tweet from disasters
    Returns:
        clean_tokens: tokenized message
    """
    
    # adjusting text
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # Create tokens 
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    
    # Lemmatise words
    clean_tokens = [WordNetLemmatizer().lemmatize(t) for t in tokens]

    return clean_tokens


def build_model():
    """
    Build model with a pipeline
    """

    # create pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([('text', Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                                                     ('tfidf', TfidfTransformer()),
                                                     ])),
                                  ('length', Pipeline([('count', FunctionTransformer(compute_text_length, validate=False))]))]
                                 )),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    # use GridSearch to tune model with optimal parameters
    parameters = {'features__text__vect__ngram_range':[(1,2),(2,2)],
            'clf__estimator__n_estimators':[50, 100]
             }
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, n_jobs=4, verbose=3)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """   
    Input:
        model: Model trained
        X_test: Dataframe, validation data for model
        Y_test: Dataframe, actual labels for the test data in X
        category_names: categories to be evaluated
    Returns:
        None: Prints out report
    """
    y_pred = model.predict(X_test)

    print (classification_report(Y_test, y_pred, target_names=category_names))
    print (accuracy_score(Y_test, y_pred))


def save_model(model, model_filepath):
    """
    Saves model as a pickle file to model_filepath
    
    Input:
        model: model to be pickled
        model_filepath: filepath where model will be saved
    Returns:
        None: Pickle file will be created at model_path
    """
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))    


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

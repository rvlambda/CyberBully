import joblib
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import re, string
import keras
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

# turn a doc into clean tokens
def clean_doc(doc, vocab):
    # split into tokens by white space
    tokens = doc.split()
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # filter out tokens not in vocab
    tokens = [w for w in tokens if w in vocab]
    tokens = ' '.join(tokens)
    return tokens

# integer encode and pad documents
def encode_docs(tokenizer, max_length, docs):
    # integer encode
    encoded = tokenizer.texts_to_sequences(docs)
    # pad sequences
    padded = pad_sequences(encoded, maxlen=max_length, padding='post')
    return padded

# classify a review as negative or positive
def predict_sentiment(review, vocab, tokenizer, max_length, model):
    # clean review
    line = clean_doc(review, vocab)
    # encode and pad review
    padded = encode_docs(tokenizer, max_length, [line])
    # predict sentiment
    yhat = model.predict(padded, verbose=0)
    # retrieve predicted percentage and label
    percent_pos = yhat[0,0]
    if round(percent_pos) == 0:
        return 'NEGATIVE'
    return 'POSITIVE'

def read__vocab_list(filename):
    # open file
    file = open(filename, 'r', encoding="utf-8")
    # Read text
    vocab = file.read()
    # close file
    file.close()
    return vocab


def get_prediction(data,model):
    """
    Predict the class of a given data point.
    """
    result = 0.0
    vocab = read__vocab_list("Model/vocab.txt")
    vocab = vocab.split("\n")
    Tokenizer = pickle.load(open("Model/token.pickle","rb"))
    result = predict_sentiment(data,vocab,Tokenizer,256,model)
    #if result > 0.5:
    #    result = 1.0
    
    return result
import os
import pickle
from util import Counter
from ngram_model import ngrams
from preprocess import preprocess_text
from nltk.tokenize import word_tokenize


def create_model(d, f):
    model = Counter()
    for file in f:
        content = preprocess_text(d+file)
        c = ngrams(content, 2)
        model.update(c)
    return model

def save_model(model, f):
    pickle.dump(model, open(f, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

def load_model(f):
    return pickle.load(open(f, "rb"))

def rank(sentence, model):
    orig_sentence = sentence
    sentence = word_tokenize(sentence)
    model_keys = len(model.keys())
    model_values = sum(model.values())
    c = 0
    for i in range(len(sentence)-1):
        w, w_ = sentence[i], sentence[i+1]
        w, w_ = w.lower(), w_.lower()
        c += model[(w, w_)]
    print(orig_sentence, (c/len(sentence)), model_values)

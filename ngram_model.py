from nltk.tokenize import word_tokenize
from util import Counter

def ngrams(corpus, n):
    model = Counter()
    for sentence in corpus:
        sent = word_tokenize(sentence)
        for i in range(len(sent)-1):
            key = tuple(sent[i:i+n])
            key = tuple(w.lower() for w in key)
            model[key] += 1
    return model

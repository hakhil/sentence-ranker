from nltk.tokenize import sent_tokenize

def preprocess_text(file):
    with open(file, encoding='latin-1') as f:
        contents = f.read()
        contents = sent_tokenize(contents)
    return contents

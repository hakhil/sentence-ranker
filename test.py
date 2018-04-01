import os
import pickle
from ngram_model import ngrams
from nltk.tokenize import sent_tokenize, word_tokenize
from bigram_model import create_model, save_model, load_model, rank
from util import Counter
from pathlib import Path
from preprocess import preprocess_text

# Model current used is a bigram model generated from the gutenberg text corpus
if __name__ == "__main__":
    training_data_dir = "./gutenberg/"
    model_file = "./bigram_model.p"
    training_files = os.listdir(training_data_dir)

    p = Path(model_file)
    model = Counter()
    if p.exists():
        model = load_model(model_file)
    else:
        model = create_model(training_data_dir, training_files)
        save_model(model, model_file)

    # Fetch some book content for testing
    content = preprocess_text(training_data_dir+training_files[4])

    max_c = -1
    max_s = ""
    min_c = 99999999
    min_s = ""

    #content = ["and the"]
    for i in range(len(content)):
        score = rank(content[i], model)
        #print(content[i], score)
        if score > max_c and len(content[i].split(" ")) > 1:
            max_c = score
            max_s = content[i]
        if score < min_c and len(content[i].split(" ")) > 1:
            min_c = score
            min_s = content[i]
    print("Max:", max_c, max_s)
    print("Min:", min_c, min_s)

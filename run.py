import os
import pickle
from tkinter import *
from ngram_model import ngrams
from nltk.tokenize import sent_tokenize, word_tokenize
from bigram_model import create_model, save_model, load_model, rank
from util import Counter
from pathlib import Path
from preprocess import preprocess_text

class EditPanel(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.S = Scrollbar(parent)
        self.T = Text(parent, height=20, width=100)
        self.S.pack(side=RIGHT, fill=Y)
        self.T.pack(side=LEFT, fill=Y)
        self.S.config(command=self.T.yview)
        self.T.config(yscrollcommand=self.S.set)
        self.T.pack()
        self.T.insert(END, "Default window")

        # Event binding to text panel
        self.T.bind('<Key>', key_event_handler)

class CanvasPanel(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.S = Scrollbar(parent)
        canvas_width = 600
        canvas_height = 350
        self.C = Canvas(parent, width=canvas_width, height=canvas_height)
        self.S.pack(side=RIGHT, fill=Y)
        self.C.pack(side=LEFT, fill=Y)
        self.S.config(command=self.C.yview)
        self.C.pack()

def key_event_handler(event):
    x_indent = 20
    y_indent = 20
    text = event.widget.get("1.0", "end-1c")
    text = text.split("\n")
    res = ""
    for i, s in enumerate(text):
        score = rank(s, model)
        res += str(i) + ") " + s + " : " + str(score) + "\n"
    text = res
    cp.C.delete("all")
    cp.C.create_text(x_indent, y_indent, width=560, fill="darkblue", font="Times 15", anchor=NW, text=text)

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

    # Start editor and canvas windows
    root = Tk()
    ep = EditPanel(root)
    ep.pack(fill="both", expand=True)

    top = Toplevel(root)
    cp = CanvasPanel(top)
    cp.pack(fill="both", expand=True)

    root.mainloop()

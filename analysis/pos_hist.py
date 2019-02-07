import os
from collections import Counter

import spacy

DATA_DIR = "data"
DATA_FILES = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".txt")]

nlp = spacy.load("en")

for data_file in DATA_FILES:
    with open(data_file, encoding="utf-8") as f:
        pos_hist = Counter(token.pos_ for doc in nlp.pipe(f) for token in doc)
    print(data_file, *pos_hist.most_common(), sep="\t")

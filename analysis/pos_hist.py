import os
from collections import Counter

import spacy

DATA_DIR = "data"
DATA_FILES = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".txt")]

nlp = spacy.load("en")
all_pos = set()
hists = {}
for data_file in DATA_FILES:
    with open(data_file, encoding="utf-8") as f:
        pos_hist = Counter(token.pos_ for doc in nlp.pipe(f) for token in doc)
    all_pos.update(pos_hist)
    hists[data_file] = pos_hist

all_pos = sorted(all_pos)
print("", *all_pos, sep="\t")
for data_file, pos_hist in sorted(hists.items()):
    print(os.path.basename(os.path.splitext(data_file)[0]), *[pos_hist[p] for p in all_pos], sep="\t")

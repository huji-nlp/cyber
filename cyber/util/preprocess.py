import os

import spacy
from spacy.symbols import ADJ, ADV, NOUN, PROPN, VERB

from cyber.util.split_data import DATA_SUBDIRS, clean_file_path

nlp = spacy.load("en")

CONTENT_POS = {ADJ, ADV, NOUN, PROPN, VERB}  # TODO , X, NUM}


def mask(tok, drop=False, content=False, func=False):
    assert not (drop and content and func), "Cannot drop all tokens"
    is_content = tok.pos in CONTENT_POS
    if is_content and content or not is_content and func:
        if drop:
            return ""
        else:
            return " " + tok.pos_ + " "
    else:
        return tok.text_with_ws


def print_doc(file, **kwargs):
    s = "".join(mask(t, **kwargs) for t in doc)
    print(" ".join(s.split()).strip(), file=file)


if __name__ == "__main__":
    for subdir in DATA_SUBDIRS:
        for div in ("train", "validation", "test"):
            file_path = clean_file_path(subdir, div)
            splitext = os.path.splitext(file_path)
            with open(file_path, encoding="utf-8") as f, \
                    open(splitext[0] + ".dropcontent" + splitext[1], "w", encoding="utf-8") as dropcontent, \
                    open(splitext[0] + ".poscontent" + splitext[1], "w", encoding="utf-8") as poscontent, \
                    open(splitext[0] + ".dropfunc" + splitext[1], "w", encoding="utf-8") as dropfunc, \
                    open(splitext[0] + ".posfunc" + splitext[1], "w", encoding="utf-8") as posfunc, \
                    open(splitext[0] + ".pos" + splitext[1], "w", encoding="utf-8") as pos:
                for doc in nlp.pipe(map(str.strip, f)):
                    print_doc(dropcontent, drop=True,  content=True)
                    print_doc(poscontent,  drop=False, content=True)
                    print_doc(dropfunc,    drop=True,  func=True)
                    print_doc(posfunc,     drop=False, func=True)
                    print_doc(pos,         drop=False, content=True, func=True)

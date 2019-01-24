import contextlib
import os
import re
from itertools import groupby

from unidecode import unidecode

# Indicate we should stop reading this file
STOP = (
    "PGP PUBLIC KEY BLOCK",  # Encryption key
    "Kommentar-Feed", "Warenkorb"  # German
)


def read_file(path):
    with open(path, "rb") as f:
        for line in f.read().decode("utf-8", errors="ignore").splitlines():
            line = line.strip()
            # skip list of references or encryption keys at the end
            if line.startswith(('References', 'PGP')) or any(s in line for s in STOP):
                break
            m = re.match(r"## LANG (.*)", line)  # skip non-english documents
            if m:
                lang = m.group(1)
                if lang != "en":
                    break
            if not line.startswith("##"):  # skip URL header
                yield line


def new_line(s):
    if re.match(r"(.?\s*\[\d+\]\s*)+", s) or \
            not re.search(r"[a-zA-Z]", s.replace("GMT", "")):  # button list or non alphabetic
        return None
    m = re.match(r".?\s*(\(\d+\)|\d+\.)", s)
    if m:
        return m.group(1)  # other enumerated list
    return bool(s)


def clean(text):
    # remove urls
    text = re.sub(r'\b(?:(?:https?|ftp|file):/+)+\S*', ' ', text)
    text = re.sub(r'\b(?:(?:https?|ftp|file):\b/)+\S*', ' ', text)

    # remove pictures/html
    text = re.sub(r'\b(?:(\.jpg|\.png|\.JPG|\.PNG|\.html))', ' ', text)

    # remove non-ascii characters
    # text = ''.join(char for char in text if ord(char) < 128)

    # others
    text = text.replace('xcxbb', ' ')
    text = text.replace('xexac', ' ')
    text = text.replace('xex', ' ')
    # text = text.replace('xc2',' ')
    text = text.replace('xbb', ' ')
    # text = re.sub(r'\b(x\d\d|x\D\d|x\d\D)', ' ', text)
    text = re.sub(r'\b(d=|fbid=)+\S*', ' ', text)
    # text = text.replace('xe2',' ')
    # text = text.replace('x82',' ')
    text = text.replace('xac', ' ')
    text = text.replace('//', ' ')
    text = text.replace('[', ' ')
    text = text.replace(']', ' ')
    text = text.replace('*', ' ')
    text = text.replace('#', ' ')
    text = text.replace('\\', ' ')
    text = text.replace('(BUTTON)', ' ')
    text = re.sub(r"\b(buy now|(add )?to cart|comments feed)\b", ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'1_ X\b', ' ', text)
    text = re.sub(r'javascript:\S*', ' ', text)
    text = re.sub(r'BUTTON\s*(Input)?\s*(\(not)?\s*(implemented\))?', ' ', text)
    text = text.replace('_', ' ')
    text = text.replace('13abJg9Rc2uRgWN7NLRmeM5Q1jkh7wfcMh', '')
    text = text.replace('c607a2f21680e3777808d3f320e551ab', '')
    # text = text.replace('x99','')
    # text = text.replace('x94','')
    # text = text.replace('x93','')
    # text = text.replace('xa1','')
    # text = text.replace('x80','')
    text = text.replace('~', '')
    text = text.replace('+', '')
    text = text.replace('xb1ol', '')
    text = text.replace('0 item(s)', '')
    text = re.sub(r'Powered by+\S*', '', text)
    text = re.sub(r'\b(?:(\.onion))', ' ', text)
    text = re.sub(r'File:+\S*', '', text)
    text = re.sub(r'\./ucp+S*', '', text)

    # remove consecutive spaces
    text = re.sub(r'[\s=]+', ' ', text)
    text = re.sub(r'--+', '--', text)
    text = re.sub(r'(\.\s*\.)+', '..', text)

    return text.strip()


def clean_lines(all_lines):
    # noinspection PyTypeChecker
    for non_empty, lines in groupby(all_lines, key=new_line):  # group multiple newlines together
        if non_empty:
            line = clean(unidecode(" ".join(lines)))
            if len(line.split()) > 3:  # skip three-word or shorter lines
                yield line


@contextlib.contextmanager
def dummy():
    yield None


def detect_duplicates(s):
    return re.sub(r"\d", "", s.lower())


def clean_directory(dirname, print_files=True):
    output = []
    out_dir = dirname + "_clean"
    if print_files:
        os.makedirs(out_dir, exist_ok=True)
    for filename in os.listdir(dirname):
        with open(os.path.join(out_dir, filename), 'w', encoding="utf-8") if print_files else dummy() as f_out:
            # remove duplicate lines
            for _, lines in groupby(clean_lines(read_file(os.path.join(dirname, filename))), key=detect_duplicates):
                for line in lines:
                    if print_files:
                        print(line, file=f_out)
                    else:
                        output.append(line)
                    break
    return None if print_files else output


if __name__ == "__main__":
    clean_directory(os.getcwd())

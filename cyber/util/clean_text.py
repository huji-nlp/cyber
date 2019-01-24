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


DELETE_ROW_PATTERNS = (r'View the latest post [a-zA-Z0-9,: ]* [ap]m', r'(asked|edited) \d* \w* ago in \d*', r'\d{6,}',
                       r'PGP SIGNATURE')
DELETE_PATTERN1 = (
    r'\b(?:(?:https?|ftp|file):/+)+\S*', r'\b(?:(?:https?|ftp|file):\b/)+\S*',  # remove urls
    r'\b(?:(\.jpg|\.png|\.JPG|\.PNG|\.html))',  # remove pictures/html
    r'\b(d=|fbid=)+\S*', r'\(\d+ points\)',
)
DELETE_WORD1 = ('xcxbb', 'xexac', 'xex', 'xbb', 'xac', '//', '[', ']', '*', '#', '\\', '(BUTTON)')
DELETE_PATTERN_IGNORECASE = (
    r"\b(buy now|(add )?to cart|comments feed)\b",
)
DELETE_PATTERN2 = (
    r'1_ X\b', r'javascript:\S*', r'BUTTON\s*(Input)?\s*(\(not)?\s*(implemented\))?',
)
DELETE_WORD2 = (
    '_', '13abJg9Rc2uRgWN7NLRmeM5Q1jkh7wfcMh', 'c607a2f21680e3777808d3f320e551ab', '~', '+', 'xb1ol', '0 item(s)',
)
DELETE_PATTERN3 = (
    r'Powered by+\S*', r'\b(?:(\.onion))', r'File:+\S*', r'\./ucp+S*',
)


def clean(text):
    for pattern in DELETE_ROW_PATTERNS:
        if re.search(pattern, text):
            return ""
    # remove non-ascii characters
    # text = ''.join(char for char in text if ord(char) < 128)
    for pattern in DELETE_PATTERN1:
        text = re.sub(pattern, ' ', text)
    for word in DELETE_WORD1:
        text = text.replace(word, ' ')
    for pattern in DELETE_PATTERN_IGNORECASE:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    for pattern in DELETE_PATTERN2:
        text = re.sub(pattern, ' ', text)
    for word in DELETE_WORD2:
        text = text.replace(word, ' ')
    for pattern in DELETE_PATTERN3:
        text = re.sub(pattern, ' ', text)

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

import re
import os


def clean(text):
    # remove urls
    text = re.sub(r'\b(?:(?:https?|ftp|file):/+)+\S*', ' ', text)
    text = re.sub(r'\b(?:(?:https?|ftp|file):\b/)+\S*', ' ', text)

    # remove pictures/html
    text = re.sub(r'\b(?:(\.jpg|\.png|\.JPG|\.PNG|\.html))', ' ', text)

    # remove numbers in brackets
    text = re.sub(r'(\[\d\])', '', text)
    text = re.sub(r'(\[\d\d\])', '', text)
    text = re.sub(r'(\[\d\d\d\])', '', text)
    text = re.sub(r'(\[\d\d\d\d\])', '', text)

    # remove non-ascii characters
    # text = ''.join(char for char in text if ord(char) < 128)

    # others
    text = text.replace('xcxbb', ' ')
    text = text.replace('xexac', ' ')
    text = text.replace('xex', ' ')
    # text = text.replace('xc2',' ')
    text = text.replace('xbb', ' ')
    text = re.sub(r'\b(x\d\d|x\D\d|x\d\D)', ' ', text)
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
    text = text.replace('_', ' ')
    text = text.replace('13abJg9Rc2uRgWN7NLRmeM5Q1jkh7wfcMh', '')
    # text = text.replace('x99','')
    # text = text.replace('x94','')
    # text = text.replace('x93','')
    # text = text.replace('xa1','')
    # text = text.replace('x80','')
    text = text.replace('~', '')
    text = text.replace('+', '')
    text = text.replace('xb1ol', '')
    return text


for filename in os.listdir(os.getcwd()):
    with open(filename, encoding="utf-8") as f_in, open(filename + '.clean', 'w', encoding="utf-8") as f_out:
        for line in f_in:
            string_new = clean(line.strip())
            if string_new == '.':
                string_new = ''
            if ('References' or 'PGP') in string_new:
                break
            print(string_new, file=f_out)

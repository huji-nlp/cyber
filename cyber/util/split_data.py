import os

from cyber.util.clean_text import clean_directory

DATA_DIR = "data"
DATA_SUBDIRS = (
    ("ebay",),
    ("onion", "legal"),
    ("onion", "illegal"),
)

TRAIN_RATIO = .8
VALIDATION_RATIO = .1
MAX_LENGTH = 9999


def clean_file_path(subdir, div):
    return os.path.join(DATA_DIR, "%s.%s.clean.txt" % ("_".join(subdir), div))


def split_data(subdirs):
    clean = {subdir: clean_directory(os.path.join(DATA_DIR, *subdir), print_files=False) for subdir in subdirs}
    min_num = min(map(len, clean.values()))
    train = int(TRAIN_RATIO * min_num)
    validation = train + int(VALIDATION_RATIO * min_num)
    for subdir, lines in clean.items():
        for div, start, end in ("train", 0, train), ("validation", train, validation), ("test", validation, min_num):
            file_path = clean_file_path(subdir, div)
            with open(file_path, "w", encoding="utf-8") as f:
                for line in lines[start:end]:
                    print(line[:MAX_LENGTH], file=f)
            print("Created '%s' with %d lines" % (file_path, end - start + 1))


if __name__ == "__main__":
    split_data(DATA_SUBDIRS)

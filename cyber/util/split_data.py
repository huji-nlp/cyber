import os
from collections import Counter
from random import sample, seed

import numpy as np

from cyber.util.clean_text import clean_directory, detect_duplicates

# sys.path.append("/cs/snapless/oabend/borgr/cyber")

DATA_DIR = "data"
DATA_SUBDIRS = (
    ("ebay",),
    ("onion", "drugs", "legal"),
    ("onion", "drugs", "illegal"),
    ("onion", "forums", "legal"),
    ("onion", "forums", "illegal"),
)

TRAIN_RATIO = .8
VALIDATION_RATIO = .1
MAX_LENGTH = 1400


def clean_file_path(subdir, div):
    os.makedirs(os.path.join(DATA_DIR, div), exist_ok=True)
    return os.path.join(DATA_DIR, div, "_".join(subdir) + ".txt")


def get_clean_lines(subdir):
    lines_set = set()
    lines = []
    deleted = Counter()
    for l in clean_directory(os.path.join(DATA_DIR, "raw", *subdir), print_files=False):
        key = detect_duplicates(l)
        if len(l) <= MAX_LENGTH and key not in lines_set:
            lines.append(l)
            lines_set.add(key)
        else:
            deleted.update([l])
    print("%s: %d filtered duplicates, duplicity histogram: %s" % (
        os.path.join(*subdir), len(deleted), sorted(Counter(deleted.values()).items())))
    return lines


def split_data(subdirs):
    clean = {subdir: get_clean_lines(subdir) for subdir in subdirs}
    min_num = min(map(len, clean.values()))
    train = int(TRAIN_RATIO * min_num)
    validation = train + int(VALIDATION_RATIO * min_num)
    for subdir, lines in clean.items():
        print("%s: sampling %d out of %d instances" % ("_".join(subdir), min_num, len(lines)))
        line_nums = sorted(sample(list(range(len(lines))), min_num))
        lines = np.array(lines)[line_nums]
        len_hist = Counter()
        for div, start, end in ("train", 0, train), ("validation", train, validation), ("test", validation, min_num):
            file_path = clean_file_path(subdir, div)
            with open(file_path, "w", encoding="utf-8") as f:
                for line in lines[start:end]:
                    len_hist[len(line)] += 1
                    print(line[:MAX_LENGTH], file=f)
            print("Created '%s' with %d lines, %d lines exceed maximum length, mean length in characters: %.1f, "
                  "maximum observed length: %d" % (file_path, end - start + 1,
                                                   len([l for l in len_hist.elements() if l > MAX_LENGTH]),
                                                   np.mean(list(len_hist.elements())).item(),
                                                   max(len_hist.elements())))


if __name__ == "__main__":
    np.random.seed(0)
    seed(0)
    split_data(DATA_SUBDIRS)

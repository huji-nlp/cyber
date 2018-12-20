from clean_text import clean_directory

for subdir in "data/ebay", "data/onion/legal", "data/onion/illegal":
    clean_directory(subdir)

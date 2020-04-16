# The Language of Legal and Illegal Activity on the Darknet

This repository contains code and data for the following [paper](https://www.aclweb.org/anthology/P19-1419.pdf):

    @inproceedings{choshen-etal-2019-language,
        title = "The Language of Legal and Illegal Activity on the {D}arknet",
        author = "Choshen, Leshem  and
          Eldad, Dan  and
          Hershcovich, Daniel  and
          Sulem, Elior  and
          Abend, Omri",
        booktitle = "Proceedings of the 57th Conference of the Association for Computational Linguistics",
        month = jul,
        year = "2019",
        address = "Florence, Italy",
        publisher = "Association for Computational Linguistics",
        url = "https://www.aclweb.org/anthology/P19-1419",
        pages = "4271--4279"
    }

## Contents

* `csvs`: Onion labels (e.g., legal/illegal) per website
* `cyber`: code to read and classify documents
* `ebay`: documents from eBay (product descriptions)
* `ebay_clean`: documents from eBay (product descriptions), after cleaning
* `experiments`: AllenNLP configuration files
* `onion`: documents from Onion (website text), classified by label
* `onion_clean`: documents from Onion, classified by label, after cleaning
* `paper`: source code for the paper

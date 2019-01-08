from scipy import spatial, stats
# from scipy.spatial import jensenshannon
import random
import itertools
import os
from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('classic')

sns.set()
sns.set_style("dark")


def jensenshannon(p, q, base=None):
    """
    Compute the Jensen-Shannon distance (metric) between
    two 1-D probability arrays. This is the square root
    of the Jensen-Shannon divergence.
    The Jensen-Shannon distance between two probability
    vectors `p` and `q` is defined as,
    .. math::
       \\sqrt{\\frac{D(p \\parallel m) + D(q \\parallel m)}{2}}
    where :math:`m` is the pointwise mean of :math:`p` and :math:`q`
    and :math:`D` is the Kullback-Leibler divergence.
    This routine will normalize `p` and `q` if they don't sum to 1.0.
    Parameters
    ----------
    p : (N,) array_like
        left probability vector
    q : (N,) array_like
        right probability vector
    base : double, optional
        the base of the logarithm used to compute the output
        if not given, then the routine uses the default base of
        scipy.stats.entropy.
    Returns
    -------
    js : double
        The Jensen-Shannon distance between `p` and `q`
    .. versionadded:: 1.2.0
    Examples
    --------
    >>> from scipy.spatial import distance
    >>> distance.jensenshannon([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 2.0)
    1.0
    >>> distance.jensenshannon([1.0, 0.0], [0.5, 0.5])
    0.46450140402245893
    >>> distance.jensenshannon([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    0.0
    """
    # p = np.asarray(p)
    # q = np.asarray(q)
    # p = p / np.sum(p, axis=0)
    # q = q / np.sum(q, axis=0)
    # m = (p + q) / 2.0
    # left = spatial.distance.rel_entr(p, m)
    # right = spatial.distance.rel_entr(q, m)
    # js = np.sum(left, axis=0) + np.sum(right, axis=0)
    # if base is not None:
    #     js /= np.log(base)
    # return np.sqrt(js / 2.0)
    # def jensenshannon(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    p = p / np.sum(p, axis=0)
    q = q / np.sum(q, axis=0)
    m = (p + q) / 2
    left = stats.entropy(p, m)
    right = stats.entropy(q, m)
    js = np.sum(left, axis=0) + np.sum(right, axis=0)
    if base is not None:
        js /= np.log(base)
    return np.sqrt(js / 2.0)

    # return (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2


def compare_domains(domains, names):

    words = list(set([key for domain in domains for key in domain]))
    word_nums = [sum(val for val in domain.values()) for domain in domains]
    print("word counts:", ",".join([str(name) + ":" + str(num) for num, name in zip(word_nums, names)]))
    percentages = [[domain[key] / word_num for key in words]
                   for word_num, domain in zip(word_nums, domains)]

    series_dict = {name:{} for name in names}
    for ind1, ind2 in itertools.combinations(list(range(len(percentages))), 2):
        row1 = np.array(percentages[ind1])
        row2 = np.array(percentages[ind2])
        # find cosine similarity (not distance)
        cosine = 1 - spatial.distance.cosine(row1, row2)
        # print("cosine between", names[ind1], "and", names[ind2], "is", cosine)
        wassesrstein = stats.wasserstein_distance(row1, row2)
        # print("Wasserstein(EMD) between", names[
        #       ind1], "and", names[ind2], "is", wassesrstein)
        dkl = stats.entropy(row1, row2)
        # print("kl divergence between", names[
        #       ind1], "and", names[ind2], "is", dkl)
        variational = sum((np.abs(row1 - row2)))
        print("l1 between", names[ind1], "and", names[ind2], "is", variational)
        js = jensenshannon(row1, row2)
        print("js between", names[ind1], "and", names[ind2], "is", js)
        series_dict[names[ind1]][names[ind2]] = (js, variational)
        series_dict[names[ind2]][names[ind1]] = (js, variational)
    df = pd.DataFrame.from_dict(series_dict)
    df = df.reindex(sorted(df.columns), axis=1)
    df = df.reindex(sorted(df.columns), axis=0)
    df.to_csv("domain_comparison.csv")
    print(df)


def files_itr(dr):
    for root, dirs, filenames in os.walk(dr):
        for filename in filenames:
            yield os.path.join(root, filename)


def count_words(itr):
    word_count = Counter()
    for path in itr:
        with open(path) as fl:
            for line in fl:
                if (not line.startswith("URL")) and (not line.startswith("LANG")):
                    word_count.update(line.split())
    return word_count


def split_randomly(dr, parts=2):
    paths = [[] for i in range(parts)]
    sums = [0,0]
    for root, dirs, filenames in os.walk(dr):
        for filename in filenames:
            assert len(paths) == parts, len(paths)
            part = random.randrange(0, parts)
            paths[part].append(os.path.join(root, filename))
    return paths

data_dir = "data/"
names = ["all onion", "illegal", "legal", "ebay"]
# dirs = ["onion_clean", "onion_clean/illegal_clean",
#         "onion_clean/legal_clean", "ebay_clean"]
# domains = [count_words(files_itr(os.path.join(data_dir, dr))) for dr in dirs]
legal_filenames = ["/cs/snapless/oabend/borgr/cyber/data/ebay.half2.clean.txt", "/cs/snapless/oabend/borgr/cyber/data/ebay.half.clean.txt"]
illegal_filenames = ["/cs/snapless/oabend/borgr/cyber/data/onion_illegal.half2.clean.txt", "/cs/snapless/oabend/borgr/cyber/data/onion_illegal.half.clean.txt"]
ebay_filenames = ["/cs/snapless/oabend/borgr/cyber/data/onion_legal.half2.clean.txt", "/cs/snapless/oabend/borgr/cyber/data/onion_legal.half.clean.txt"]
filename_lists = [legal_filenames + illegal_filenames, illegal_filenames, legal_filenames, ebay_filenames]
domains = [count_words(filenames) for filenames in filename_lists]
half_onions = split_randomly(os.path.join(data_dir,"onion_clean"))
domains += [count_words(itr) for itr in [[filename for filename in filename_list if "half." in filename] for filename_list in filename_lists]]
half_names = [name + "_half1" for name in names]
domains += [count_words(itr) for itr in [[filename for filename in filename_list if "half2" in filename] for filename_list in filename_lists]]
half2_names = [name + "_half2" for name in names]
names += half_names
names += half2_names

compare_domains(domains, names)

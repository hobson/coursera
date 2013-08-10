#!/usr/bin/env python

from collections import defaultdict, Counter

# count[0] = 1-gram counts
# count[1] = 2-gram counts
count = 
count = defaultdict(int)
count.update(fish)


def get_words(s):
    # only alpha words, no numbers
    splitter = re.compile(r'\W*')
    # filter after map, unlike collective intelligence which reverses this
    return set(filter(filter_fun, map(get_words.map_fun, splitter.splitter(s))))
get_words.map_fun = lower
get_words.max_len = 16
get_words.min_len = 3
get_words.filter_fun = lambda x: len(x) >= get_words.min_len and len(x) <= get_words.max_len

fish_count = Counter({'carp': 10, 'eel': 1, 'perch': 3, 'trout': 1, 'salmon': 1, 'perch': 3, 'whitefish': 2})

count = [Counter(get_words(s))]

class Classifier:
    def __init__(self, get_features=get_words, s='', path=''):
        self.get_features = get_features
        # TODO: use a defaultdict everywhere a dict is used
        self.feature_count = Counter()
        self.category_count = defaultdict(Counter)  # dict of Counters, count of documents binned in each category 
        self.num_items = 0

    def increment_feature(self, feature, category):
        "Increase the count for a feature<->category association"
        feature_count[feature] += Counter((feature,))
        # self.feature_count.setdefault(feature, Counter())
        # self.feature_count[feature].setdefault(category, 0)
        # self.feature_count[feature][category] += 1

    def increment_count(self, category):
        "Increase the count of documents associated with a category"
        self.category_count.setdefault(category, 0)
        self.category_count[category] += 1

    def count_feature_in_category(self, feature, category):
        "Count of feature occurences in a category"
        # using a defaultdict everywhere would take care of this
        if feature in self.feature_count and category in feature_count(feature):
            return float(self.feature_count[feature][category])
        return 0.

    def items_in_category(self, category):
        if category in self.category_count:
            return float(self.category_count[category])    
        return 0.

    def num_items(self):
        "Total of the number of items that have been classified"
        # inefficient?, so cache this, if possible to detect with a dict has changed efficiently
        return sum(self.category_count.values())

    def categories(self):
        return list(self.category_count)

    def train(self, item, category):
        features = self.get_features(item)
        feature_count[feature] += Counter(features)


N_len = max(c for w, c in count.iteritems())
N = [0] * N_len + 1
for w, c in count.iteritems():
    N[c] += 1
N_total = sum(N)

V = len(set(k for k in count if isinstance(k, basestring)))

# stupid backoff method, needs example code

# from Jurafsky, Smoothing: Add-One lecture, 2:00/06:30
def mle_estimate(word, prior_word):
    """given a prior word, estimate the probability of the next (current) word"""
    float(count[(prior_word, word)]) / count[(prior_word,)]


# "Errata" in Jurafsky, Smoothing: Add-One lecture, 2:00/06:30, doesn't mention what V means (vocabulary size)
def laplace_estimate(word, prior_word, k=1, V=V):
    float(count[(prior_word, word)] + k) / (count[(prior_word,)] + k * V)


def interpolation_estimate(word, prior_word, m=1, V=V):
    float(count[(prior_word, word)] + m * (1./V)) / (count[(prior_word,)] + m)


def unigram_prior(word, prior_word, m=1, V=V):
    float(count[(prior_word, word)] + m * count(prior_word) / float(V)) / (count[prior_word] + m)

c_star = defaultdict(lambda: N[1] / N_total)

for w, c in count.iteritems():
    c_star[w] = (count(w) + 1) * N[c + 1] / N[c]


# word should be ngram and they should all be tuples, so that backoff can just remove the first and take the remainder (or nothing)
def good_turing(word, prior_word, m=1):
    c_star[word]
    



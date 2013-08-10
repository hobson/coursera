#!/usr/bin/env python

from collections import defaultdict, Counter
import re


def get_words(s):
    """Identify the features (words) in a string

    TODO: make this a generator to improve memory efficiency
    """
    # only alpha words, no numbers
    splitter = re.compile(r'\W*')
    # filter after map, unlike collective intelligence which reverses this
    return filter(get_words.filter_fun, map(get_words.map_fun, splitter.split(s)))
get_words.map_fun = str.lower
get_words.max_len = 16
get_words.min_len = 3
get_words.filter_fun = lambda x: len(x) >= get_words.min_len and len(x) <= get_words.max_len


class Basic:
    def __init__(self, get_features=get_words, s='', path=''):
        self.get_features = get_features
        # {feature: total_num_occurrences}
        self.feature_count = defaultdict(Counter)
        # {category: {feature: num_occurences,...}}
        self.category_count = Counter()  # dict of Counters, count of documents binned in each category 
        # self.num_items = 0

    def increment_feature_count(self, feature, category):
        "Increase the count for a feature<->category association"
        # if isinstance(feature, (list, )):   # don't do tuple, or hasasstr(__iter__) in case feature = N-gram stored as tuple
        #     self.feature_count[category] += Counter(feature)
        # self.feature_count.setdefault(feature, Counter())
        # self.feature_count[feature].setdefault(category, 0)
        # self.feature_count[feature][category] += 1
        self.feature_count[category][feature] += 1

    def increment_category_count(self, category):
        "Increase the count of documents associated with a category"
        # self.category_count.setdefault(category, 0)
        self.category_count[category] += 1

    def count_feature_in_category(self, feature, category):
        "Count of feature occurences in a category"
        # using a defaultdict everywhere would take care of this
        #if category in self.feature_count and feature in self.feature_count[category]:
        return float(self.feature_count[category][feature])
        
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

    def train(self, string, category):
        """Identify features (N-grams or words) in a string and associate that string with a category
        >>> c = Basic()
        >>> c.train('The quick brown fox jumps over the lazy dog', 'good')
        >>> c.train('Make quick money in the online casino', 'bad')
        >>> c.count_feature_in_category('quick', 'good')
        1.0
        >>> c.count_feature_in_category('quick', 'bad')
        1.0
        >>> c.num_items()
        2
        """
        self.feature_count[category] += Counter(self.get_features(string))
        self.category_count[category] += 1



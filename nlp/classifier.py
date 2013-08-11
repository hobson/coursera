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
    def __init__(self, get_features=get_words, s='', path='', threshold=None):
        self.get_features = get_features
        # {category: {feature: num_occurences,...}}
        self.feature_count = defaultdict(Counter)
        # {category: total_num_occurrences}
        self.category_count = Counter()  # dict of Counters, count of documents binned in each category 
        self.threshold = defaultdict(lambda: 1.)
        if threshold:
            self.threshold.update(threshold)
        # self.num_items = 0

    def feature_probability(self, feature, category):
        return float(self.feature_count[category][feature]) / self.category_count[category]

    def weighted_feature_probability(self, feature, category, weight=1., assumed_probability=.5):
        """
        Use an assumed probability (for 0 occurence features) and weight it to produce a pseudo-probability
        >>> import examples as e
        >>> cl = Basic()
        >>> cl.train(e.training_set)
        >>> cl.weighted_feature_probability('quick', 'good')
        0.625
        >>> cl.weighted_feature_probability('money', 'good')
        0.25
        >>> cl.train(e.training_set)
        >>> cl.weighted_feature_probability('money', 'good')  # doctest: +ELLIPSIS
        0.1666...
        """

        total_feature_count = sum((self.feature_count[c][feature] for c in self.category_count))
        return  (weight * assumed_probability + total_feature_count * self.feature_probability(feature, category)) / (weight + total_feature_count)

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

    def num_categories(self):
        "Total of the number of items that have been classified"
         # inefficient?, so cache this, if possible to detect with a dict has changed efficiently
        return len(self.category_count)

    def categories(self):
        return list(self.category_count)

    def train(self, string, category=None):
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
        if isinstance(string, basestring):
            self.feature_count[category] += Counter(self.get_features(string))
            self.category_count[category] += 1
        else:
            for cat, strings in string.iteritems():
                for s in strings:
                    self.train(s, cat)

    def __repr__(self):
        return repr(self.feature_count)

    def __str__(self):
        return str(self.feature_count)


class NaiveBayes(Basic):

    def repeated_item_probability(self, item, category):
        item_feature_count = Counter(get_words(item))
        p = 1.
        for f, c in item_feature_count.iteritems():
            # this doesn't make sense unless you're trying to compare the probabilities for numerouse vs single occurences of a feature
            p *= pow(self.weighted_feature_probability(f, category), c)
        return p
    
    def item_probability(self, item, category):
        item_features = set(get_words(item))
        p = 1.
        for f in item_features:
            p *= self.weighted_feature_probability(f, category)
        return p
    
    def category_probability(self, item, category):
        prior_category_probability = float(self.category_count[category]) / self.num_categories()
        return self.item_probability(item, category) * prior_category_probability

    def classify(self, item, default=None):
        max_prob = 0.
        next_prob = 0.
        best_match = default
        for category, count in self.category_count.iteritems():
            prob = self.item_probability(item, category)
            if prob > max_prob:
                next_prob = max_prob
                max_prob = prob
                if max_prob > next_prob * self.threshold:
                    best_match = category
        return best_match


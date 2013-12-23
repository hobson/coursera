import re
import nltk

def strip_HTML(s):
    """Silly HTML stripper"""
    result = ''
    total = 0
    for c in s:
        if c == '<':
            total = 1
        elif c == '>':
            total = 0
            result += ' '
        elif total == 0:
            result += c
    return result


def get_words(s, regex='[^\'_-.a-zA-Z0-9]+|\W+\s+'):
    if isinstance(regex, basestring):
        regex = re.compile(regex)
    return regex.split(s)


def get_sentences(s, regex='[.?!](\W+)|'):
    if isinstance(regex, basestring):
        regex = re.compile(regex)
    return regex.split(s)

#!/usr/bin/env bash
# segment, sort, and count words using nothing more than a bash oneliner

# jurafsky's version that does word frequency analysis on an ascii file containing the works of Shakespear
# tr 'A-Z' 'a-z' < shakes.txt | tr -sc 'A-Za-z' '\n' | sort | uniq -c | sort -n -r | less

cat $1 | tr [:upper:] [:lower:] | tr -sc 'A-Za-z' '\n' | sort | uniq -c | sort -n -r | less

# use nlp.classifier.Basic
# training it with each class being an ID (or title) of an article
# def count_article_words():
#     words = {}
#     article_words = {}


def article_words_matrix(words, article_words):
    interesting_words = []
    for w, c in words:
        if 3 < c < len(article_words) * 0.6:
            interesting_words.append(w)
    return [[(word in aw and aw[word] or 0) for word in interesting_words] for aw in article_words]



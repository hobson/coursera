import re
import fnmatch
import os
import datetime
import pytz

from collections import Mapping, Counter

import dateutil.parser
try:
    from scipy.sparse import csr_matrix  # compressed, sparse, row-wise (slow column slicing)
except:
    csr_matrix = None

from pug.nlp.util import get_words


#import nltk



def get_file_paths(root='.', pattern=None):
    matches = []
    pattern = pattern or '*'
    for root, dirnames, filenames in os.walk('root'):
        for filename in fnmatch.filter(filenames, '*.c'):
            matches.append(os.path.join(root, filename))



# use nlp.classifier.Basic
# training it with each class being an ID (or title) of an article
# def count_article_words():
#     words = {}
#     article_words = {}


def document_words(words, article_words):
    interesting_words = []
    for w, c in words:
        if 3 < c < len(article_words) * 0.6:
            interesting_words.append(w)
    return [[(word in aw and aw[word] or 0) for word in interesting_words] for aw in article_words]


class Occurences(object):
    """Word/string occurrence matrix, words in rows, documents in columns
    """

    def __init__(self, matrix=None, words=None):
        #self.N = 1000000  # 1 million words should keep collisions to a minimum
        self.words = ('',)
        self.word_indices = {'': 0}
        self.matrix = csr_matrix([[0]])

        if matrix:
            if words and len(words) == len(matrix):
                self.words = tuple(words)
            else:  # if matrix is a dict instead of an numpy.ndarray (matrix) of counts
                self.words = tuple(k for k in matrix)
            self.word_indices = dict((w, i) for i, w in enumerate(self.words))
            self.matrix = csr_matrix(matrix)

        # words are first index (row index), documents/passages/sequences of words are columns
        #self.matrix = self.dictify(matrix)

    # def dictify(self, matrix):
    #     d = {}
    #     for k, row in enumerate(matrix):
    #         if not any(row):
    #             continue
    #         # if matrix is a list of dicts of counts
    #         if isinstance(row, Mapping):
    #             for k, v  in enumerate(row.iteritems()):
    #                 d[self.N * self.word_indices[k]] = v
    #         # if matrix is a list of lists of counts
    #         else:
    #             for k, count in enumerate(row):
    #                 d[k*self.N + ]
    #     return d

    def append_document(self, doc, wordlist=None):
        """Add a column to the occurence matrix representing the word counts in a single document"""
        if isinstance(doc, basestring):
            doc = Counter(get_words(doc))
        if not isinstance(doc, Mapping) and len(doc) <= len(wordlist):
            doc = dict(zip(doc, wordlist[:len(doc)]))
        iptr = self.matrix.indptr.tolist()
        num_docs = self.matrix.shape[1]
        self.matrix = csr_matrix((self.matrix.data, self.matrix.indices, iptr + [iptr[-1]]),
                                 shape=(len(self.words), num_docs + 1), dtype='int64')
        for word, count in doc.iteritems():
            self.matrix[self.word_index(word), num_docs] = count

    def word_index(self, word):
        if word not in self.word_indices:
            self.append_word(word)
        return self.word_indices[word]

    def append_word(self, word):
        self.word_indices[word] = len(self.words)
        self.words += [word]
        # expand the occurrence matrix by one empty row
        # todo: figure out how to extend np.array objects efficiently instead of converting to lists and appending
        iptr = self.matrix.indptr.tolist()
        self.matrix = csr_matrix((self.matrix.data, self.matrix.indices, iptr + [iptr[-1]]),
                                 shape=(len(self.words), self.num_docs), dtype='int64')

    def __repr__(self):
        return 'Occurences(%s, %s)' % (self.matrix.todense(), self.words)

# MONTHS = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
# MONTH_PREFIXES = [m[:3] for m in MONTHS]
# MONTH_SUFFIXES = [m[3:] for m in MONTHS]
# SUFFIX_LETTERS = ''.join(set(''.join(MONTH_SUFFIXES)))

RE_MONTH_NAME = re.compile('(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[acbeihmlosruty]*', re.IGNORECASE)


def make_tz_aware(dt, tz='UTC'):
    """Add timezone information to a datetime object, only if it is naive."""
    tz = dt.tzinfo or tz
    try:
        tz = pytz.timezone(tz)
    except AttributeError:
        pass
    return tz.localize(dt)


def clean_wiki_datetime(dt, squelch=False):
    """
    >>> clean_wiki_datetime([u'11 January', u'2014', u'17', u'54'])
    datetime.datetime(2014, 1, 11, 17, 54) 
    >>> clean_wiki_datetime([u'8 January', u'2014', u'00', u'46'])
    datetime.datetime(2014, 1, 8, 0, 46)
    """
    if isinstance(dt, datetime.datetime):
        return dt
    elif not isinstance(dt, basestring):
        dt = ' '.join(dt)
    try:
        return make_tz_aware(dateutil.parser.parse(dt))
    except:
        print("Failed to parse %r" % dt)
    dt = [s.strip() for s in dt.split(' ')]
    # get rid of any " at " or empty strings
    dt = [s for s in dt if s and s.lower() != 'at']

    # deal with the absence of :'s in wikipedia datetime strings

    if RE_MONTH_NAME.match(dt[0]) or RE_MONTH_NAME.match(dt[1]):
        if len(dt) >= 5:
            dt = dt[:-2] + [dt[-2].strip(':') + ':' + dt[-1].strip(':')]
            return clean_wiki_datetime(' '.join(dt))
        elif len(dt) == 4 and (len(dt[3]) == 4 or len(dt[0]) == 4):
            dt[:-1] + ['00']
            return clean_wiki_datetime(' '.join(dt))
    elif RE_MONTH_NAME.match(dt[-2]) or RE_MONTH_NAME.match(dt[-3]):
        if len(dt) >= 5:
            dt = [dt[0].strip(':') + ':' + dt[1].strip(':')] + dt[2:]
            return clean_wiki_datetime(' '.join(dt))
        elif len(dt) == 4 and (len(dt[-1]) == 4 or len(dt[-3]) == 4):
            dt = [dt[0], '00'] + dt[1:]
            return clean_wiki_datetime(' '.join(dt))

    try:
        return make_tz_aware(dateutil.parser.parse(' '.join(dt)))
    except Exception as e:
        if squelch:
            from traceback import format_exc
            print format_exc(e) +  '\n^^^ Exception caught ^^^\nWARN: Failed to parse datetime string %r\n      from list of strings %r' % (' '.join(dt), dt)
            return dt
        raise(e)
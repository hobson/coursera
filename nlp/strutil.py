import re
import fnmatch
import os
from collections import Mapping, Counter
try:
    from scipy.sparse import csr_matrix  # compressed, sparse, row-wise (slow column slicing)
except:
    csr_matrix = None

import string
import dateutil.parser

PUNC = unicode(string.punctuation)

#import nltk


def strip_HTML(s):
    """Simple, clumsy, slow HTML tag stripper"""
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


def strip_edge_punc(s, punc=PUNC):
    if not isinstance(s, basestring):
        return [strip_edge_punc(unicode(s0), punc) for s0 in s]
    return s.strip(punc)


WORD_SPLIT_IGNORE_EXTERNAL_APOSTROPHIES = re.compile('\W*\s\'{1,3}|\'{1,3}\W+|[^-\'_.a-zA-Z0-9]+|\W+\s+')
WORD_SPLIT_PERMISSIVE = re.compile('[^-\'_.a-zA-Z0-9]+|[^\'a-zA-Z0-9]\s\W*')
SENTENCE_SPLIT = re.compile('[.?!](\W+)|$')


# this regex assumes "s' " is the end of a possessive word and not the end of an inner quotation, e.g. He said, "She called me 'Hoss'!"
def get_words(s, splitter_regex=WORD_SPLIT_IGNORE_EXTERNAL_APOSTROPHIES, 
              preprocessor=strip_HTML, postprocessor=strip_edge_punc, blacklist=None, whitelist=None):
    r"""Segment words (tokens), returning a list of all tokens (but not the separators/punctuation)

    >>> get_words('He said, "She called me \'Hoss\'!". I didn\'t hear.')
    ['He', 'said', 'She', 'called', 'me', 'Hoss', 'I', "didn't", 'hear.']
    >>> get_words('The foxes\' oh-so-tiny den was 2empty!')
    ['The', 'foxes', 'oh-so-tiny', den', 'was', '2empty']
    """
    postprocessor = postprocessor or unicode
    preprocessor = preprocessor or unicode
    blacklist = blacklist or get_words.blacklist
    whitelist = whitelist or get_words.whitelist
    try:
        s = open(s, 'r')
    except:
        pass
    try:
        s = s.read()
    except:
        pass
    if not isinstance(s, basestring):
        try:
            # flatten the list of lists of words from each obj (file or string)
            return [word for obj in s for word in get_words(obj)]
        except:
            pass
    try:
        s = preprocessor(s)
    except:
        pass
    if isinstance(splitter_regex, basestring):
        splitter_regex = re.compile(splitter_regex)
    words = map(postprocessor, splitter_regex.split(s))
    if isinstance(whitelist, (list, tuple, set)):
        return [word for word in words if word in whitelist and word not in blacklist]
    return [word for word in words if word not in blacklist]
get_words.blacklist = ('', None, '\'', '.', '_', '-')
get_words.whitelist = None


def get_file_paths(root='.', pattern=None):
    matches = []
    pattern = pattern or '*'
    for root, dirnames, filenames in os.walk('root'):
        for filename in fnmatch.filter(filenames, '*.c'):
            matches.append(os.path.join(root, filename))


def get_sentences(s, regex=SENTENCE_SPLIT):
    if isinstance(regex, basestring):
        regex = re.compile(regex)
    return [sent for sent in regex.split(s) if sent]


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


def clean_wiki_datetime(dt):
    """
    >>> clean_wiki_datetime([u'11 January', u'2014', u'17', u'54'])
    datetime.datetime(2014, 1, 11, 17, 54) 
    >>> clean_wiki_datetime([u'8 January', u'2014', u'00', u'46'])
    datetime.datetime(2014, 1, 8, 0, 46)
    """
    at_i = None
    for i, s in enumerate(dt):
        if s.endswith('at'):
            dt[i] = dt[i][:-3]
            at_i = i
    if at_i is not None and len(dt) > (at_i + 1) and dt[at_i + 1] and not ':' in dt[at_i + 1]:
        dt = dt[:at_i] + [dt[at_i + 1].strip() + ':' + dt[at_i + 2].strip()]
    elif len(dt) == 4:
        dt = dt[:2] + [dt[2] + ':' + dt[3]]
    ans = ' '.join([s.strip() for s in dt])
    try:
        return dateutil.parser.parse(ans)
    except Exception as e:
        from traceback import format_exc
        print format_exc(e) +  '\n^^^ Exception caught ^^^\nWARN: Failed to parse datetime string %r\n      from list of strings %r' % (ans, dt)
        return ans

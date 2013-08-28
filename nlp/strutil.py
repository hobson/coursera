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


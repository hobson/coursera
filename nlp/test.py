

def assert_this(expression="1 + 1 = 2", prefix="ERROR evaluating this expression:\n", eval_context=locals(), suffix=''):
    "Evaluation context isn't being passed properly so this function doesn't work."
    print eval_context.__dict__
    assert(eval(expression), prefix + expression + suffix, eval_context)


def main():
    import examples as ex
    import classifier as cl

    nb = cl.NaiveBayes()
    nb.train(ex.training_set)
    assert ".15624 < nb.unique_item_probabilty('quick rabbit', 'good') < .15626", "Collective intelligence example, loc 4275/11262, NaiveBayes probability of a document/string with only two words."
    assert "0.049 < nb.unique_item_probabilty('quick rabbit', 'bad') < 0.051", "Collective intelligence example, loc 4275/11262, NaiveBayes probability of a document/string with only two words."

    from doctest import testmod
    testmod(cl)


if __name__ == '__main__':
    from sys import exit
    exit(main())

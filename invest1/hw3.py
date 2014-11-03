# hw3.py
if __name__ == '__main__':
    import sim
    close = sim.price_dataframe(symbols='sp5002012', price_type='close')
    actual = sim.price_dataframe(symbols='sp5002012', price_type='actual_close')
    below5 = actual < 5.0
    below5diff = below5.diff()
    print "Number of events found: {0}", sum(sum(below5diff))




def is_num(s):
    try:
        return float(s)
    except ValueError:
        return None

def load(path, verbosity=0):
    """
    Read a csv file from the specified path, return a dict of lists or list of lists (according to `format`)

    path: csv file path
    
    >>> table = load('../data/ex1data.txt')
    >>> len(table)
    3
    >>> [len(table[k]) for k in table]
    [97, 97]
    """
    import csv
    from progressbar import ProgressBar
    from collections import OrderedDict
    if not path:
        return
    if format:
        format = format[0].lower()
    table = None
    
    # see http://stackoverflow.com/a/4169762/623735 if considering using format='rU' without the 'b'
    with open(path, 'rUb') as fpin:  # U = universal EOL reader, b = binary
        csvr = csv.reader(fpin, dialect=csv.excel)

        if not headers:
            while not header or not any(headers):
                headers = csvr.next()

        if all(is_num(h) for h in headers):
            row = headers
            headers = chr(ord('A') + i % 26) * (int(i / 26) + 1) for i in range(len(headers))
            table = OrderedDict([headers[i], [row[i]] for i in range(len(headers))]])
            row_num = 1
        else
            table = OrderedDict([header, [] for header in headers])
            row_num = 0

        file_size = os.fstat(fpin.fileno()).st_size

        pbar = None
        if 2 > verbosity > 0 :
            pbar = ProgressBar(maxval=file_size)
            pbar.start()

        eof = False
        while csvr and not eof:
            if pbar:
                pbar.update(fpin.tell())
            rownum += 1
            row = []
            row_dict = OrderedDict()
            # skips rows with all empty strings as values,
            while not row or not any(len(x) for x in row):
                try:
                    row = csvr.next()
                    if verbosity > 1:
                        logger.info('  row content: ' + repr(row))
                except StopIteration:
                    eof = True
                    break
            if eof:
                break
            if numbers:
                # try to convert the type to a numerical scalar type (int, float etc)
                row = [tryconvert(v, empty=None, default=None) for v in row]
            if row:
                N = min(max(len(row), 0), len(norm_names))  # truncate the table width
                for k, v in zip(headers[:N], row[:N]):
                    table[k] += [v]
        if file_len > fpin.tell():
            logger.warn("Only %d of %d bytes were read." % (fpin.tell(), file_len))
        if pbar:
            pbar.finish()
    return table


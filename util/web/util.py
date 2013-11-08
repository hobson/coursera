NULL_VALUES = (None, 'None', 'none', '<None>', 'Null', 'null', '<Null>', 'N/A', 'n/a')
NAN_VALUES = (float('inf'), 'INF', 'inf', '+inf', '+INF', float('nan'), 'nan', 'NAN', float('-inf'), '-INF', '-inf')
BLANK_VALUES = ('', ' ', '\t', '\n', '\r', ',')

def representation(model, field_names=None):
    """
    Unicode representation of a particular model instance (object or record or DB table row)
    """
    if field_names is None:
        all_names = model._meta.get_all_field_names()
        field_names = getattr(model, 'IMPORTANT_FIELDS', ['pk'] + all_names[:min(representation.default_fields, len(all_names))])
    retval = model.__class__.__name__ + u'('
    retval += ', '.join("%s" % (repr(getattr(model, s, '') or '')) for s in field_names[:min(len(field_names), representation.max_fields)])
    return retval + u')'
representation.max_fields = 10
representation.default_fields = 3

def listify(values, N=1, delim=None):
    """Return an N-length list, with elements values, extrapolating as necessary.

    >>> listify("don't split into characters")
    ["don't split into characters"]
    >>> listify("len = 3", 3)
    ['len = 3', 'len = 3', 'len = 3']
    >>> listify(0, 2)
    [0, 0]
    """
    ans = [] if values is None else values

    # convert non-string non-list iterables into a list
    if hasattr(ans, '__iter__') and not isinstance(values, basestring):
        ans = list(ans)
    else:
        # split the string (if possible)
        if isinstance(delim, basestring):
            try:
                ans = ans.split(delim)
            except:
                ans = [ans]
        else:
            ans = [ans]

    # pad the end of the list if a length has been specified
    if len(ans):
        if len(ans) < N and N > 1:
            ans += [ans[-1]] * (N - len(ans))
    else:
        if N > 1:
            ans = [[]] * N

    return ans

def intify(obj):
    try:
        return int(float(obj))
    except:
        try:
            return ord(str(obj)[0].lower())
        except:
            try:
                return len(obj)
            except:
                try:
                    return hash(str(obj))
                except:
                    return 0

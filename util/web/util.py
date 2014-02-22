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



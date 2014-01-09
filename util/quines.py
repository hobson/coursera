#!/usr/bin/env python
"""Quines: short self-replicating python scripts""" 

# cribbed from wikipedia:
s = 's = %r\nprint(s%%s)'
print(s%s)

# one-liner based on wikipedia, but less pythonic
s = 's = %r;print(s%%s)';print(s%s)

# compressed, minimal characters and lines? (less pythonic)
s='s=%r;print(s%%s)';print(s%s)

# based on wikipedia quine for python, but updated to use new string formatting, still not pythonic
x='x={!r}r;print(x.format(x))';print(x.format(x))
# or
s = 's={!r}r;print(s.format(s))';print(s.format(s))

# shouldn't we add the shabang to make self-replicating
#!/usr/bin/env python
s = '#!/usr/bin/env python\ns={!r};print(s.format(s))';print(s.format(s))

# bash idea that is invalid (not allowed to read from disk)
# cat $0

#
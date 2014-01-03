#!/usr/bin/env python

"""Revert file permissions

Have you ever accidentally changed the file permissions on a git working copy?
This command line script will revert all such changes.

>>> git diff | git_revert  # doctest: +ELLIPSIS
File permissions reverted according to the supplied diff. ... files were affected.
"""

import sys
import re
import os

OLD_MODE = re.compile(r'')
def main():
    for line in sys.stdin:
        match = OLD_MODE.match(line)
        if match:
            match.group(1)

        if fn and om and nm:
            os.chmod(fn, om)
        

if __name__ == "__main__":
    main()

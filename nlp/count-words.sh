#!/usr/bin/env bash
# segment, sort, and count words using nothing more than a bash oneliner

# jurafsky's version that does word frequency analysis on an ascii file containing the works of Shakespear
# tr 'A-Z' 'a-z' < shakes.txt | tr -sc 'A-Za-z' '\n' | sort | uniq -c | sort -n -r | less

cat $1 | tr [:upper:] [:lower:] | tr -sc 'A-Za-z' '\n' | sort | uniq -c | sort -n -r | less

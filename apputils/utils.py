from heapq import merge
from itertools import groupby
from operator import itemgetter


def list_inner_join(a:list,b:list):
    d = {}
    for row in b:
        d[row[0]] = row
    for row_a in a:
        row_b = d.get(row_a[0])
        if row_b is not None:  # join
            yield row_a + row_b[1:]
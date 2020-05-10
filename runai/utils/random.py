from __future__ import absolute_import # to support 'import random' in Python 2

import random as r
import string as s

number = lambda a=10, b=100: r.randint(a, b)

def string(length=5):
    return ''.join(r.choice(s.ascii_letters + s.digits + s.punctuation) for _ in range(length))

def strings(count=number(2, 10)):
    return [string() for _ in range(count)]

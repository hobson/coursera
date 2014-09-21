#!/usr/bin/env python

"""Cryptography utilities for Coursera Cryptography Class"""


class OTP(object):
    
    def __init__(self, key=None, ishex=True):
        if isinstance(key, bytearray):
            self.key = str(key).decode('hex')
        elif ishex:
            self.key = key.decode('hex')
        else:
            self.key = key

    def encrypt(self, msg):
        cypher = []
        for i, c in enumerate(msg):
            # ^ = bitwise XOR (not exponentiation)
            cypher.append(chr(ord(c) ^ ord(self.key[i])))
        return ''.join(cypher)

    def decrypt(self, msg):
        return self.encrypt(msg)

    def __repr__(self):
        return 'OTP(%r, ishex=True)' % self.key.encode('hex')


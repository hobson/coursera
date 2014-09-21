"""Hash collision prediction and production

>>> x2 = 'F'*32
>>> y1 = '0'*31 + '1'
>>> y2 = '0'*32
>>> x1 = D(y1, hexxor(E(y2, x2), hexxor(y2, y1)))
>>> f1(x1, y1) == f1(x2, y2)
True
>>> x1, y1
('1c56781937f4ccb1f81052af67ec7db8', '00000000000000000000000000000001')
>>> x2, y2
('FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF', '00000000000000000000000000000000')
>>> y3 = '0'*32
>>> x3 = '0'*31 + '3'
>>> x4 = '0'*31 + '4'
>>> f2(x3, y3) == f2(x4, y4)
True
>>> x3, y3
('00000000000000000000000000000003', '00000000000000000000000000000000')
>>> x4, y4
('00000000000000000000000000000004', '9620be31c2af468342168205e92b60ad')
"""


from random import randrange

def experiment(N=365, M=2):
    bds = {}
    T = 0
    bd = randrange(N)
    while True:
        T += 1
        if bd in bds:
            bds[bd] += 1
        else:
            bds[bd] = 1
        if bds[bd] >= M:
            break
        bd = randrange(N)
    return T


BLOCK_SIZE, KEY_SIZE = 16, 16
from Crypto.Cipher import AES
AES.block_size=BLOCK_SIZE

from hexstr import hexxor

def E(x=None, y=None, block_size=BLOCK_SIZE, key_size=KEY_SIZE):
    x = x or ('0'*key_size*2)
    y = y or ('0'*block_size*2)
    cipher = AES.new(x.decode('hex'), AES.MODE_ECB, ('0'*block_size).decode('hex'))
    return cipher.encrypt(y.decode('hex')).encode('hex')

def D(x=None, y=None, block_size=BLOCK_SIZE, key_size=KEY_SIZE):
    x = x or ('0'*key_size*2)
    y = y or ('0'*block_size*2)
    cipher = AES.new(x.decode('hex'), AES.MODE_ECB, ('0'*block_size).decode('hex'))
    return cipher.decrypt(y.decode('hex')).encode('hex')

def f1(x, y, hex=True):
    if hex:
        return hexxor(E(y, x), y)

def f2(x, y, hex=True):
    if hex:
        return hexxor(E(x, x), y)




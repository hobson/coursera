import re
import urllib
#from PyCrypto import AES
import hexstr

"""
The CBC decryption algorthm is 
m[i] = [XOR(IV, D(k, c[0])] + [XOR(c[i-1], D(k, c[i])) for i in range(1, len(c))] 
"""

def website(url=r'http://crypto-class.appspot.com/po?er=f20bdba6ff29eed7b046d1df9fb7000058b1ffb4210a580f748b4ac714c001bd4a61044426fb515dad3f21f18aa577c0bdf302936266926ff37dbf7035d5eeb4',
        mode='CBC',
        get_regex=r'\w*[?][a-zA-Z]+[=]',
        block_size=16):
    rest_getter = re.search(get_regex, url).group()
    print rest_getter
    base_url, c = url.split(rest_getter)
    print base_url, c
    print len(c.decode('hex'))
    # discard the last block?  don't need to do this, just following example first
    c = c.decode('hex')[:-block_size]
    print len(c)
    N = len(c)
    m = ['_'] * N
    for j in range(block_size, N):
        print j, '='*50
        prefix = c[:N-j]
        suffix = c[N-j+1:]
        print 'trying: ' + prefix.encode('hex') + '[]'*j + suffix.encode('hex')
        for i in range(256):
            print i
            g = chr(i)
            gxorpad = hexstr.strxor(g + ''.join(m[N-j+block_size : N]), chr(j % block_size) * (j - block_size + 1))
            gc = hexstr.strxor(c[N - j - 1:N-j], gxorpad)
            gc_str = c[:N - j - 1] + gc + c[-block_size:]
            response = urllib.urlopen(base_url + rest_getter + gc_str.encode('hex'))
            if padding_oracle(response):
                break
        m[N - j + block_size - 1] = g[0]
        print ''.join(m)
    return ''.join(m)


def padding_oracle(response):
    """Return True if the http response indicates a valid padding block.
    Return False if certainly not a valid pad.
    Return None or a probability 0 < p < 1 if it is uncertain whether the padding is valid
    
    For the coursea app, 404 is returned for an invalid pad,
                         403 for an invalid mac (or url)
                         500 for server fault
                         200 for a valid page load
    """
    return not (padding_oracle.edict.get(response.getcode(), None) == 'pad')
padding_oracle.edict = { 500: 'connection', 404: 'mac', 403: 'pad', 200: 'OK'}


def inject(cypher_text=r'20814804c1767293b99f1d9cab3bc3e7ac1e37bfb15599e5f40eef805488281d',
            plain_text=r'Pay Bob 100$',
            desired_plain_text=r'Pay Bob 500$',
            mode='CBC'):
    "Produce the cyphertext for the requested plain text message without know the key!"
    modified_cypher_text = cypher_text
    return modified_cypher_text
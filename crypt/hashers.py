"""crypt.hash

chained hash for incremental downloading of large files while maintaining a chained MAC

>>> chained_hash('data/birthday_video.mp4')
03c08f4ee0b576fe319338139c045c89c3e8e9409633bea29442e21425006ea8
"""

from Crypto.Hash import SHA256


def chained_sha256(path='data/birthday.mp4', block_size=1024, renew=True, prepend=False):
    blocks = []
    fp = open(path, 'rb')
    print 'Reading %s bytes from file %s' % (len(fp.read()), fp.name)  
    fp.close()
    with open(path, 'rb') as fp:
        block = True
        while block:
            block = fp.read(block_size)
            if block:
                blocks += [block]
    print 'Read %s blocks of %s bytes each, and one last block of %s bytes for a total of %s bytes.' % (
        len(blocks), block_size, len(blocks[-1]), sum(len(block) for block in blocks))
    tag = b''
    if not renew:
        sha = SHA256.new()
    for block in reversed(blocks):
        if renew:
            sha = SHA256.new()
        if prepend:
            sha.update(tag + block)
        else:
            sha.update(block + tag)
        tag = sha.digest()
    return tag.encode('hex')


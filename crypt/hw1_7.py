#hw1_7.py

from utils import OTP

k = '6c73d5240a948c86981bc294814d'
ed = OTP(k, ishex=True)
msg = 'test message'
assert(ed.decrypt(ed.encrypt(msg)) == msg)

assert(OTP(k, ishex=True).decrypt(msg)==OTP(msg.encode('hex'), ishex=True).decrypt(k))

k = ed.decrypt('attack at dawn').encode('hex')
print 'key = %r' % k
real_ed = OTP(k, ishex=True)
print 'original encrypted message, regenerated'
real_ed.encrypt("attack at dawn").encode('hex')
print 'new encrypted message'
real_ed.encrypt("attack at dusk").encode('hex')

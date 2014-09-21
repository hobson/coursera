#hw1_7.py

from utils import OTP

k = '6c73d5240a948c86981bc294814d'
ed = OTP(k, ishex=True)
msg = 'test message'
m1 = ed.encrypt(msg)
print m1.encode('hex')
m2 = ed.decrypt(msg)
print m2.encode('hex')
print ed.decrypt(m1)
assert(ed.decrypt(ed.encrypt(msg)) == msg)
kstr = k.decode('hex')[:len(msg)]
print 'message is %d long' % len(msg)
print 'key is %d long' % len(k)
assert(OTP(k, ishex=True).decrypt(msg)==OTP(msg.encode('hex'), ishex=True).decrypt(kstr))

k = ed.decrypt('attack at dawn').encode('hex')
print 'key = %r' % k
real_ed = OTP(k, ishex=True)
print 'original encrypted message, regenerated:'
print real_ed.encrypt("attack at dawn").encode('hex')
print 'new encrypted message:'
print real_ed.encrypt("attack at dusk").encode('hex')

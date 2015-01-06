#  3250  2015-01-05 11:02:26 mkdir stuff
#  3251  2015-01-05 11:02:28 cd stuff
#  3252  2015-01-05 11:02:31 curl http://challenge.shopcurbside.com
#  3253  2015-01-05 11:02:41 curl http://challenge.shopcurbside.com/start
#  3254  2015-01-05 11:03:11 curl get-session
#  3255  2015-01-05 11:03:32 curl http://challenge.shopcurbside.com/get-session
#  3256  2015-01-05 11:03:51 curl http://challenge.shopcurbside.com/get-session/400d4d25b9c540ce9d1d745231764ae4
#  3257  2015-01-05 11:04:00 curl http://challenge.shopcurbside.com/400d4d25b9c540ce9d1d745231764ae4
#  3258  2015-01-05 11:04:56 -H "Authorization: Bearer 00D50000000IehZ\!AQcAQH0dMHZfz972Szmpkb58urFRkgeBGsxL_QJWwYMfAbUeeG7c1E6
#  3259  2015-01-05 11:06:31 curl -H "Session: 400d4d25b9c540ce9d1d745231764ae4" http://challenge.shopcurbside.com/start/
#  3260  2015-01-05 11:07:00 curl -H "Session: 400d4d25b9c540ce9d1d745231764ae4" http://challenge.shopcurbside.com/start

# $ curl http://challenge.shopcurbside.com
# On the right track. You can start here: "/start"

# $ curl http://challenge.shopcurbside.com/start
# {
#   "error": "\"Session\" header is missing. \"/get-session\" to get a session id."
# }

# $ curl get-session
# curl: (6) Could not resolve host: get-session

# $ curl http://challenge.shopcurbside.com/get-session
# 400d4d25b9c540ce9d1d745231764ae4

# $ curl -H "Session: 400d4d25b9c540ce9d1d745231764ae4" http://challenge.shopcurbside.com/start
# {
#   "depth": 0, 
#   "id": "start", 
#   "message": "There is something we want to tell you, lets see if you can figure this out by finding all of our secrets", 
#   "next": [
#     "34ffe00db65f4576b5add43dda39ff99", 
#     "ebdf4d2f11514626a1b07d745d4a0fc6", 
#     "64bbc0003e824075ad59fb5cfaaac4cd"
#   ]
# }

# $ curl -H "Session: 34ffe00db65f4576b5add43dda39ff99" http://challenge.shopcurbside.com/next
# {
#   "error": "Invalid session id, a token is valid for 10 requests."
# }

# $ curl -H "Session: 34ffe00db65f4576b5add43dda39ff99" http://challenge.shopcurbside.com/start
# {
#   "error": "Invalid session id, a token is valid for 10 requests."
# }


# U+34ff is not a valid character

d = {
  "depth": 0, 
  "id": "start", 
  "message": "There is something we want to tell you, lets see if you can figure this out by finding all of our secrets", 
  "next": [
    "34ffe00db65f4576b5add43dda39ff99", 
    "ebdf4d2f11514626a1b07d745d4a0fc6", 
    "64bbc0003e824075ad59fb5cfaaac4cd"
  ]
}
s = ''
for w in d['next']:
    for c in w[0:-1:2]:
        s += c.decode("hex")
s
d['next'][0].decode("hex")
d['next'][1].decode("hex")
d['next'][2].decode("hex")
from collections import Counter
c = Counter(d['next'][0])
c = Counter(''.join(s.decode('hex') for s in d['next']))
c
c['4']
c['\xff']
c['\xad']
c = Counter(''.join(s.decode('hex').decode('unicode') for s in d['next']))
c = Counter(''.join(s.decode('hex').decode('utf8') for s in d['next']))

d = {
  "depth": 0, 
  "id": "start", 
  "message": "There is something we want to tell you, lets see if you can figure this out by finding all of our secrets", 
  "next": [
    "34ffe00db65f4576b5add43dda39ff99", 
    "ebdf4d2f11514626a1b07d745d4a0fc6", 
    "64bbc0003e824075ad59fb5cfaaac4cd"
  ]
}

next = d['next']

len(next[0])
# 32


# (predict)laneh@predict:~/src/stuff$ 
# $ curl -H "Session: 400d4d25b9c540ce9d1d745231764ae4" http://challenge.shopcurbside.com/next/34ffe00db65f4576b5add43dda39ff99
# <!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">
# <title>404 Not Found</title>
# <h1>Not Found</h1>
# <p>The requested URL was not found on the server.  If you entered the URL manually please check your spelling and try again.</p>
# (predict)laneh@predict:~/src/stuff$ 
# $ curl -H "Session: 400d4d25b9c540ce9d1d745231764ae4" http://challenge.shopcurbside.com/34ffe00db65f4576b5add43dda39ff99
# {
#   "error": "Invalid session id, a token is valid for 10 requests."
# }(predict)laneh@predict:~/src/stuff$ 
# $ curl http://challenge.shopcurbside.com/get-session
# f8b895e46abd475ea491c725765f67ab(predict)laneh@predict:~/src/stuff$ 
# $ curl -H "Session: f8b895e46abd475ea491c725765f67ab" http://challenge.shopcurbside.com/34ffe00db65f4576b5add43dda39ff99
# {
#   "depth": 1, 
#   "id": "34ffe00db65f4576b5add43dda39ff99", 
#   "next": [
#     "81dd32039f2f4bfea3ab35f40c994a0e", 
#     "adad9a3a96464d2ba2977d9863e33794", 
#     "33aaccd7d4fe40019160c49b689c7358"
#   ]
# }(predict)laneh@predict:~/src/stuff$ 
# $ curl -H "Session: f8b895e46abd475ea491c725765f67ab" http://challenge.shopcurbside.com/81dd32039f2f4bfea3ab35f40c994a0e
# {
#   "depth": 2, 
#   "id": "81dd32039f2f4bfea3ab35f40c994a0e", 
#   "next": [
#     "a2b07e1917cd4d03aa8557f0ee982e80", 
#     "998e40abd4c14993b3a06880e03a5106", 
#     "d587883b577046bcbba08841bbf5aae0", 
#     "2a5a561f61854a16a9bd2cd08ed13106"
#   ]
# }(predict)laneh@predict:~/src/stuff$ 
# $ curl -H "Session: f8b895e46abd475ea491c725765f67ab" http://challenge.shopcurbside.com/a2b07e1917cd4d03aa8557f0ee982e80
# {
#   "depth": 3, 
#   "id": "a2b07e1917cd4d03aa8557f0ee982e80", 
#   "next": [
#     "d9f9a7c4ac0a48959f0de596c91ee90b", 
#     "849e7b590222402ca8a3610c7d389b19", 
#     "4d1ff41945684f4e876b3635ab8ecb04", 
#     "c872e3bad7d34c83b5754dc8e045b99a"
#   ]
# }(predict)laneh@predict:~/src/stuff$ 
# $ curl -H "Session: f8b895e46abd475ea491c725765f67ab" http://challenge.shopcurbside.com/d9f9a7c4ac0a48959f0de596c91ee90b
# {
#   "depth": 4, 
#   "id": "d9f9a7c4ac0a48959f0de596c91ee90b", 
#   "secret": "A"
# }(predict)laneh@predict:~/src/stuff$
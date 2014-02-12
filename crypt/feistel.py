from hexstr import hexxor


def test_feistel(L_00, L_10):
    """Test to see if a 2-stage feistel network was used for DES-like encryption

    >>> L2_00 = ('5f67abaf', '7b50baab', '290b6e3a', '39155d6f')
    >>> L2_10 = ('bbe033c0', 'ac343a22', 'd6f491c5', 'eea6e3dd')
    >>> [test_feistel(L2a, L2b) for (L2a, L2b) in zip(L2_00, L2_10)]
    ['unk', 'unk', 'INSECURE', 'unk']
    """
    L_xor = hexxor(L_00, L_10)
    if all(c==L_xor[(i+1)%len(L_xor)] for i, c in enumerate(L_xor)):
        return 'INSECURE'
    return 'unk'


def valid_message_len(message, block_size=128, encryption='AES'):
    """Determines whether the proposed message is potentially a valid message based solely on message length

    >>> [int(valid_message_len(message)) for message in ('In this letter I make some remarks on a general principle relevant to enciphering in general and my machine.', \
                        'If qualified opinions incline to believe in the exponential conjecture, then I think we cannot afford not to make use of it.' \
                        'To consider the resistance of an enciphering process to being broken we should assume that at same times the enemy knows everything but the key being used and to break it needs only discover the key from this information.', \
                        'The most direct computation would be for the enemy to try all 2^r possible keys, one by one.', \
                        )]
    [0, 0, 1, 0]
    """
    if encryption.lower().strip() in ('AES', ):
        if len(message) < block_size - 24:
            return True

    



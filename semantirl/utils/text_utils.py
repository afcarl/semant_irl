
def pad(tokens, pad_tok, l):
    if len(tokens) >= l:
        return tokens[:l]
    num_pad = l-len(tokens)
    tok_copy = tokens[:]
    tok_copy.extend([pad_tok]*num_pad)
    return tok_copy

def one_hot(vocab, sentence):
    pass

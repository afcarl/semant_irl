
def pad(tokens, pad_tok, l):
    if len(tokens) >= l:
        return tokens[:l]
    num_pad = l-len(tokens)
    tok_copy = tokens[:]
    tok_copy.extend([pad_tok]*num_pad)
    return tok_copy

def pretty_print_sentence(tokens):
    """Remove PAD, START, EOS tokens """
    return [tok for tok in tokens if tok not in ['_EOS', '_PAD', '_START']]
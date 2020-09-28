import codecs
import numpy as np
from numpy import linalg as LA


def load_embeddings_from_w2vf(filename):
    print('loading w2vf...')

    wv = np.load(filename + '.npy')
    with codecs.open(filename + '.vocab', 'r', 'utf-8') as f_embed:
        vocab = f_embed.read().split()

    w2i = {w: i for i, w in enumerate(vocab)}

    return vocab, wv, w2i


def normalize(wv):
    # normalize vectors
    norms = np.apply_along_axis(LA.norm, 1, wv)
    wv = wv / norms[:, np.newaxis]
    return wv


def load_and_normalize(lang, filename, vocab, wv, w2i):
    vocab_curr, wv_curr, w2i_curr = load_embeddings_from_w2vf(filename)
    wv_curr = normalize(wv_curr)
    vocab[lang] = vocab_curr
    wv[lang] = wv_curr
    w2i[lang] = w2i_curr
    print('done')

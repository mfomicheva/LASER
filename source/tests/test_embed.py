import faiss
import numpy as np


def test_embeds_segment(sentence_encoder):
    embs = sentence_encoder.encode_sentences(['This is a test', 'This is a test'])
    assert embs.shape == (2, 1024)


def test_compute_dot_product(sentence_encoder):
    src_embs = sentence_encoder.encode_sentences(['This is a test', 'This is a test'])
    tgt_embs = src_embs
    faiss.normalize_L2(src_embs)
    faiss.normalize_L2(tgt_embs)
    print(np.einsum('ij,ij->i', src_embs, tgt_embs))  # same as np.sum(src_embs * tgt_embs, axis=1) but faster

import argparse
import sys

import faiss

from embed import EmbedLoad


def load_embeds(input_file, dimension):
    x = EmbedLoad(input_file, dimension, verbose=False)
    faiss.normalize_L2(x)
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_embeds')
    parser.add_argument('-t', '--target_embeds')
    parser.add_argument('-d', '--dim', default=1024)
    args = parser.parse_args()
    src_embeds = load_embeds(args.source_embeds, args.dim)
    tgt_embeds = load_embeds(args.target_embeds, args.dim)
    scores = []
    for s_emb, t_emb in zip(src_embeds, tgt_embeds):
        scores.append(s_emb.dot(t_emb))
    for score in scores:
        sys.stdout.write('{}\n'.format(score))


if __name__ == '__main__':
    main()

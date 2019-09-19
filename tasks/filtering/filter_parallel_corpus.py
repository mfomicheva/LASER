import argparse
import json
import faiss
import numpy as np
import os
import shutil
import sys
import tempfile

from langid.langid import LanguageIdentifier, model

assert os.environ.get('LASER'), 'Please set the enviornment variable LASER'
LASER = os.environ['LASER']

sys.path.append(LASER + '/source/lib')
sys.path.append(LASER + '/source')

from embed import buffered_read
from embed import SentenceEncoder

from text_processing import Token, BPEfastApply


ERROR_TYPES = {
    'EMPTY': 0,
    'LASER': 1,
    'LANGID': 2,
    'OVERLAP': 3,
}

LANGID = LanguageIdentifier.from_modelstring(model, norm_probs=True)


def prepare_data(infile, tmpdir, token_lang, bpe_codes, verbose=False):
    tok_fname = os.path.join(tmpdir, 'tok.%s' % token_lang)
    Token(infile,
          tok_fname,
          lang=token_lang,
          romanize=True if token_lang == 'el' else False,
          lower_case=True, gzip=False,
          verbose=verbose, over_write=False)
    infile = tok_fname
    bpe_fname = os.path.join(tmpdir, 'bpe.%s' % token_lang)
    BPEfastApply(infile,
                 bpe_fname,
                 bpe_codes,
                 verbose=verbose, over_write=False)
    return tok_fname, bpe_fname


def clean_data_discrete(src_file_tok, tgt_file_tok, src_lang, tgt_lang, langid=False, overlap=False):  # TODO: ignore strings consisting mostly of numbers
    filtered = {}
    counter = 0
    for src, tgt in zip(open(src_file_tok), open(tgt_file_tok)):
        if all_symbols(src.split()) or all_symbols(tgt.split()):
            continue
        if len(src) == 0 or len(tgt) == 0:
            filtered[counter] = ERROR_TYPES['EMPTY']
        elif langid and (wrong_language(src, src_lang) or wrong_language(tgt, tgt_lang)):
            filtered[counter] = ERROR_TYPES['LANGID']
        elif overlap and compute_overlap(src, tgt) > 0.6:
            filtered[counter] = ERROR_TYPES['OVERLAP']
        else:
            pass
        counter += 1
    return filtered


def wrong_language(seg, lang):
    pred, proba = LANGID.classify(seg)
    if proba > 0.9 and pred != lang:
        return True
    else:
        return False


def all_symbols(tokens):
    unique_tokens = set(tokens)
    abc_toks = set()
    for tok in unique_tokens:
        if all(chr.isalpha() for chr in tok):
            abc_toks.add(tok)
    if tokens and len(abc_toks) / len(unique_tokens) < 0.7:
        return True
    else:
        return False


def compute_overlap(src, tgt):
    set1 = set(src.split())
    set2 = set(tgt.split())
    if set1 and set2:
        return len(set1.intersection(set2)) / len(set1.union(set2))
    else:
        return 0.


def write_filtered_segments(src_file, tgt_file, out_pref, filtered):
    counter = 0
    out_fh = open(out_pref + '.filtered_segments', 'w')
    for src_line, tgt_line in zip(open(src_file), open(tgt_file)):
        if counter in filtered:
            out = ' ||| '.join([
                src_line.strip(),
                tgt_line.strip(),
                str(filtered[counter])
            ])
            out_fh.write('{}\n'.format(out))
        counter += 1


def write_segments_by_index(inpath, outpath, indexes):
    outf = open(outpath, 'w')
    counter = 0
    for line in open(inpath):
        if counter in indexes:
            outf.write('{}\n'.format(line.strip()))
        counter += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_file')
    parser.add_argument('--tgt_file')
    parser.add_argument('--src_lang')
    parser.add_argument('--tgt_lang')
    parser.add_argument('--encoder')
    parser.add_argument('--bpe_codes')
    parser.add_argument('--output_pref')
    parser.add_argument('--buffer_size', default=10, type=int, required=False)
    parser.add_argument('--max_tokens', default=1200, type=int, required=False)
    parser.add_argument('--cpu', default=True, type=bool, required=False)
    parser.add_argument('--tmpdir', default=None, type=str, required=False)
    parser.add_argument('--verbose', default=False, action='store_true', required=False)
    parser.add_argument('--debug', default=False, action='store_true', required=False)
    parser.add_argument('--threshold', default=0.85, type=float, required=False)
    parser.add_argument('--langid', default=False, action='store_true', required=False)
    parser.add_argument('--overlap', default=False, action='store_true', required=False)
    args = parser.parse_args()
    encoder = SentenceEncoder(
        args.encoder,
        max_tokens=args.max_tokens,
        sort_kind='quicksort',
        cpu=args.cpu
    )
    tmpdir = args.tmpdir if args.tmpdir else tempfile.mkdtemp()
    src_tok_file, src_bpe_file = prepare_data(args.src_file, tmpdir, args.src_lang, args.bpe_codes, verbose=args.verbose)
    tgt_tok_file, tgt_bpe_file = prepare_data(args.tgt_file, tmpdir, args.tgt_lang, args.bpe_codes, verbose=args.verbose)
    filtered = clean_data_discrete(src_tok_file, tgt_tok_file, args.src_lang, args.tgt_lang, langid=args.langid, overlap=args.overlap)
    src_fh = open(src_bpe_file)
    tgt_fh = open(tgt_bpe_file)
    out_clean_idx = open(args.output_pref + '.idx_clean', 'wb')
    batch_id = 0
    for src_sents, tgt_sents in zip(
            buffered_read(src_fh, args.buffer_size), buffered_read(tgt_fh, args.buffer_size)):
        src_embs = encoder.encode_sentences(src_sents)
        tgt_embs = encoder.encode_sentences(tgt_sents)
        faiss.normalize_L2(src_embs)
        faiss.normalize_L2(tgt_embs)
        sim_scores = np.einsum('ij,ij->i', src_embs, tgt_embs)
        batch_clean_indexes = []
        for k, score in enumerate(sim_scores):
            index = len(src_sents) * batch_id + k
            if index in filtered:
                continue
            if score < args.threshold: # and not all_symbols(src_sents[k].split()) and not all_symbols(tgt_sents[k].split()):
                filtered[index] = ERROR_TYPES['LASER']
            else:
                batch_clean_indexes.append(index)
        batch_clean_indexes = np.asarray(batch_clean_indexes, dtype=int)
        batch_clean_indexes.tofile(out_clean_idx)
        batch_id += 1
        if batch_id % 100:
            print('Processed %d batches' % batch_id)

    if not args.tmpdir:
        shutil.rmtree(tmpdir)
    all_indexes_clean = np.fromfile(args.output_pref + '.idx_clean', dtype=int)
    write_segments_by_index(args.src_file, args.output_pref + '.clean.%s' % args.src_lang, all_indexes_clean)
    write_segments_by_index(args.tgt_file, args.output_pref + '.clean.%s' % args.tgt_lang, all_indexes_clean)
    json.dump(filtered, open(args.output_pref + '.idx_filtered', 'w'))

    if args.debug:
        write_filtered_segments(args.src_file, args.tgt_file, args.output_pref, filtered)

    # TODO: to train the classifier, add missing / added operation
    # TODO: positive and negative data should not be the same
    # TODO: add length ratio
    # TODO: when > 80% segments are not alpha compute character distance


if __name__ == '__main__':
    main()

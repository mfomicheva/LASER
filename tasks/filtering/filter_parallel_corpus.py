import argparse
import faiss
import langid
import numpy as np
import os
import shutil
import sys
import tempfile

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
    return bpe_fname


def detect_lang(seg):
    return langid.classify(seg)[0]


def compute_overlap(src, tgt):
    set1 = set(src.split())
    set2 = set(tgt.split())
    return len(set1.intersection(set2))/ len(set1.union(set2))


def clean_sentences(data, src_lang, tgt_lang, threshold=0.85):
    clean_indexes = []
    filtered_indexes = []
    errors = []
    for i, (src, tgt, score) in data.items():
        if len(src) == 0 or len(tgt) == 0:
            errors.append(ERROR_TYPES['EMPTY'])
            filtered_indexes.append(i)
            continue
        elif score < threshold:
            errors.append(ERROR_TYPES['LASER'])
            filtered_indexes.append(i)
            continue
        elif detect_lang(src) != src_lang or detect_lang(tgt) != tgt_lang:
            errors.append(ERROR_TYPES['LANGID'])
            filtered_indexes.append(i)
            continue
        elif compute_overlap(src, tgt) > 0.6:
            errors.append(ERROR_TYPES['OVERLAP'])
            filtered_indexes.append(i)
            continue
        else:
            clean_indexes.append(i)
    return np.asarray(clean_indexes, dtype=int), np.asarray(filtered_indexes, dtype=int), np.asarray(errors, dtype=int)


def _write_segments_by_index(src_path, tgt_path, output_pref, src_lang, tgt_lang):
    src_fh = open(output_pref + '.clean.%s' % src_lang, 'w')
    tgt_fh = open(output_pref + '.clean.%s' % tgt_lang, 'w')
    indexes = np.fromfile(output_pref + '.idx_clean', dtype=int)
    counter = 0
    for src_line, tgt_line in zip(open(src_path), open(tgt_path)):
        if counter in indexes:
            src_fh.write('{}\n'.format(src_line.strip()))
            tgt_fh.write('{}\n'.format(tgt_line.strip()))
        counter += 1
    src_fh.close()
    tgt_fh.close()


def write_filtered_segments(src_file, tgt_file, out_pref, indexes_filtered, errors):
    counter = 0
    out_fh = open(out_pref + '.filtered_segments', 'w')
    for src_line, tgt_line in zip(open(src_file), open(tgt_file)):
        if counter in indexes_filtered:
            out = ' ||| '.join([
                src_line.strip(),
                tgt_line.strip(),
                str(errors[counter])
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
    args = parser.parse_args()
    encoder = SentenceEncoder(
        args.encoder,
        max_tokens=args.max_tokens,
        sort_kind='quicksort',
        cpu=args.cpu
    )
    tmpdir = args.tmpdir if args.tmpdir else tempfile.mkdtemp()
    src_bpe_file = prepare_data(args.src_file, tmpdir, args.src_lang, args.bpe_codes, verbose=args.verbose)
    tgt_bpe_file = prepare_data(args.tgt_file, tmpdir, args.tgt_lang, args.bpe_codes, verbose=args.verbose)
    src_fh = open(src_bpe_file)
    tgt_fh = open(tgt_bpe_file)
    out_clean_idx = open(args.output_pref + '.idx_clean', 'wb')
    out_filtered_idx = open(args.output_pref + '.idx_filtered', 'wb')
    out_errors = open(args.output_pref + '.errors', 'wb')
    batch_id = 0
    for src_sents, tgt_sents in zip(
            buffered_read(src_fh, args.buffer_size), buffered_read(tgt_fh, args.buffer_size)):
        src_embs = encoder.encode_sentences(src_sents)
        tgt_embs = encoder.encode_sentences(tgt_sents)
        faiss.normalize_L2(src_embs)
        faiss.normalize_L2(tgt_embs)
        sim_scores = np.einsum('ij,ij->i', src_embs, tgt_embs)
        indexed_data = {
            len(src_sents) * batch_id + k: (src_sents[k], tgt_sents[k], sim_scores[k]) for k in range(len(src_sents))
        }
        clean_indexes, filtered_indexes, errors = clean_sentences(indexed_data, args.src_lang, args.tgt_lang)
        clean_indexes.tofile(out_clean_idx)
        filtered_indexes.tofile(out_filtered_idx)
        errors.tofile(out_errors)
        batch_id += 1
        if batch_id % 100:
            print('Processed %d batches' % batch_id)

    if not args.tmpdir:
        shutil.rmtree(tmpdir)
    all_indexes_clean = np.fromfile(args.output_pref + '.idx_clean', dtype=int)
    write_segments_by_index(args.src_file, args.output_pref + '.clean.%s' % args.src_lang, all_indexes_clean)
    write_segments_by_index(args.tgt_file, args.output_pref + '.clean.%s' % args.tgt_lang, all_indexes_clean)
    if args.debug:
        all_indexes_filtered = np.fromfile(args.output_pref + '.idx_filtered', dtype=int)
        all_errors = np.fromfile(args.output_pref + '.errors', dtype=int)
        write_filtered_segments(args.src_file, args.tgt_file, args.output_pref, all_indexes_filtered, all_errors)

    # TODO: to train the classifier, add missing / added operation
    # TODO: positive and negative data should not be the same
    # TODO: add length ratio


if __name__ == '__main__':
    main()

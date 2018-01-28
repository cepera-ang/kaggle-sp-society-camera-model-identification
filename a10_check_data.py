# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import hashlib

from a00_common_functions import *


def md5_from_file(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def find_duplicates(path):
    files = glob.glob(path + '**/*.jpg', recursive=True)
    out = open(OUTPUT_PATH + 'files_hash_stat.csv', 'w')
    out.write('path,md5\n')
    print('Files found: {}'.format(len(files)))
    all_hashes = dict()
    for f in files:
        # print('Go for {}'.format(f))
        hsh = md5_from_file(f)
        out.write(f + ',' + hsh + '\n')
        if hsh in all_hashes:
            all_hashes[hsh].append(f)
        else:
            all_hashes[hsh] = [f]
    out.close()

    duplicate_count = 0
    for el in all_hashes:
        if len(all_hashes[el]) > 1:
            print('Duplicate found. Count: {}. List below'.format(len(all_hashes[el])))
            for item in all_hashes[el]:
                print(item)
            duplicate_count += (len(all_hashes[el]) - 1)
    print('Total duplicates found: {}'.format(duplicate_count))


if __name__ == '__main__':
    find_duplicates(INPUT_PATH + 'external/')

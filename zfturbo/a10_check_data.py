# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

from zfturbo.a00_common_functions import *
import shutil
import hashlib


def md5_from_file(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def find_duplicates(path):
    files = glob.glob(path + '**/*.jpg')
    out = open(OUTPUT_PATH + 'files_hash_stat.csv', 'w')
    out.write('path,md5\n')
    print('Files found: {}'.format(len(files)))
    all_hashes = dict()
    for f in files:
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
            print('Duplicate found. count: {}: {}'.format(len(all_hashes[el]), all_hashes[el][0]))
            duplicate_count += (len(all_hashes[el]) - 1)
    print('Duplicate found: {}'.format(duplicate_count))


if __name__ == '__main__':
    find_duplicates(INPUT_PATH + 'external/')

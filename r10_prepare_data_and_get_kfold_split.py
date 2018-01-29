# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import hashlib
import exifread
import shutil
import os
import glob
import pickle
import gzip
import numpy as np
from sklearn.model_selection import KFold


INPUT_PATH = '../input/'
OUTPUT_PATH = 'data/'


def save_in_file(arr, file_name):
    pickle.dump(arr, gzip.open(file_name, 'wb+', compresslevel=3))


def load_from_file(file_name):
    return pickle.load(gzip.open(file_name, 'rb'))


def md5_from_file(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def prepare_external_dataset(raw_path, output_path):
    exif_dict = {
        'HTC One': 'HTC-1-M7',
        'HTC6500LVW': 'HTC-1-M7',
        'HTCONE': 'HTC-1-M7',

        'Nexus 5X': 'LG-Nexus-5x',

        'XT1080': 'Motorola-Droid-Maxx',
        'XT1060': 'Motorola-Droid-Maxx',

        'Nexus 6': 'Motorola-Nexus-6',

        'XT1096': 'Motorola-X',
        'XT1092': 'Motorola-X',
        'XT1095': 'Motorola-X',
        'XT1097': 'Motorola-X',
        'XT1093': 'Motorola-X',

        'SAMSUNG-SM-N900A': 'Samsung-Galaxy-Note3',
        'SM-N9005': 'Samsung-Galaxy-Note3',
        'SM-N900P': 'Samsung-Galaxy-Note3',

        'SCH-I545': 'Samsung-Galaxy-S4',
        'GT-I9505': 'Samsung-Galaxy-S4',
        'SPH-L720': 'Samsung-Galaxy-S4',

        'NEX-7': 'Sony-NEX-7',

        'iPhone 4S': 'iPhone-4s',

        'iPhone 6': 'iPhone-6',
        'iPhone 6 Plus': 'iPhone-6',
    }

    hash_checker = dict()
    files = glob.glob(raw_path + '**/*.jpg', recursive=True)
    if os.path.isdir(output_path):
        print('Folder "{}" already exists! You must delete it before proceed!'.format(output_path))
        exit()
    os.mkdir(output_path)
    print('Files found: {}'.format(len(files)))
    for f in files:
        tags = exifread.process_file(open(f, 'rb'))
        try:
            model = str(tags['Image Model'])
        except:
            print('Broken Image Model EXIF: {}'.format(f))
            continue
        if model not in exif_dict:
            print('Skip EXIF {}'.format(model))
            continue

        # Check unique hash
        hsh = md5_from_file(f)
        if hsh in hash_checker:
            print('Hash {} for file {} alread exists. Skip file!'.format(hsh, f))
            continue
        hash_checker[hsh] = 1

        out_folder = output_path + exif_dict[model]
        if not os.path.isdir(out_folder):
            os.mkdir(out_folder)

        shutil.copy2(f, out_folder)

    copied_files = glob.glob(output_path + '**/*.jpg', recursive=True)
    print('Files in external folder: {}'.format(len(copied_files)))
    return exif_dict


def get_kfold_split(num_folds=4, cache_path=None):
    if cache_path is None:
        cache_path = OUTPUT_PATH + 'kfold_split_{}.pklz'.format(num_folds)

    if not os.path.isfile(cache_path):
        files = glob.glob(os.path.join(INPUT_PATH, 'train/*/*.jpg')) + \
              glob.glob(os.path.join(INPUT_PATH, 'external/*/*.jpg'))

        kf = KFold(n_splits=num_folds, shuffle=True, random_state=66)
        files = np.array(files)
        ret = []
        for train_index, test_index in kf.split(range(len(files))):
            train_files = files[train_index]
            test_files = files[test_index]
            ret.append((train_files, test_files))
        save_in_file(ret, cache_path)
    else:
        ret = load_from_file(cache_path)

    # check all files exists
    if 1:
        files = list(ret[0][0]) + list(ret[0][1])
        print('Files in KFold split: {}'.format(len(files)))
        for f in files:
            if not os.path.isfile(f):
                print('File {} is absent!'.format(f))
                exit()

    return ret


if __name__ == '__main__':
    # 1st param - location of your directories like 'flickr1', 'val_images' etc
    # 2nd parameter - location where files will be copied. Warning: you need to have sufficient space
    prepare_external_dataset(INPUT_PATH + 'raw/', INPUT_PATH + 'external/')

    # will return list of lists [[train1, valid1], [train2, valid2] , ... [trainK, validK]]
    kf = get_kfold_split(num_folds=4)

# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import numpy as np
import gzip
import pickle
import os
import glob
import time
import cv2
import datetime
import pandas as pd
from sklearn.metrics import fbeta_score
from sklearn.model_selection import KFold, train_test_split
from collections import Counter, defaultdict
import random
import shutil
import operator

random.seed(2016)
np.random.seed(2016)


INPUT_PATH = '../input/'
OUTPUT_PATH = '../modified_data/'
if not os.path.isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
MODELS_PATH = '../models/'
if not os.path.isdir(MODELS_PATH):
    os.mkdir(MODELS_PATH)
HISTORY_FOLDER_PATH = "../models/history/"
if not os.path.isdir(HISTORY_FOLDER_PATH):
    os.mkdir(HISTORY_FOLDER_PATH)
SUBM_PATH = '../subm/'
if not os.path.isdir(SUBM_PATH):
    os.mkdir(SUBM_PATH)


CLASSES = [
    'HTC-1-M7',
    'iPhone-6',
    'Motorola-Droid-Maxx',
    'Motorola-X',
    'Samsung-Galaxy-S4',
    'iPhone-4s',
    'LG-Nexus-5x',
    'Motorola-Nexus-6',
    'Samsung-Galaxy-Note3',
    'Sony-NEX-7']
N_CLASSES = len(CLASSES)

EXTRA_CLASSES = [
    'htc_m7',
    'iphone_6',
    'moto_maxx',
    'moto_x',
    'samsung_s4',
    'iphone_4s',
    'nexus_5x',
    'nexus_6',
    'samsung_note3',
    'sony_nex7'
]

RESOLUTIONS = {
    0: [[1520,2688]], # flips
    1: [[3264,2448]], # no flips
    2: [[2432,4320]], # flips
    3: [[3120,4160]], # flips
    4: [[4128,2322]], # no flips
    5: [[3264,2448]], # no flips
    6: [[3024,4032]], # flips
    7: [[1040,780],  # Motorola-Nexus-6 no flips
        [3088,4130], [3120,4160]], # Motorola-Nexus-6 flips
    8: [[4128,2322]], # no flips
    9: [[6000,4000]], # no flips
}

ORIENTATION_FLIP_ALLOWED = [
    True,
    False,
    True,
    True,
    False,
    False,
    True,
    True,
    False,
    False
]


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


def get_single_split(fraction=0.9, only_train=False, cache_path=None):
    if cache_path is None:
        cache_path = OUTPUT_PATH + 'single_split_{}_{}.pklz'.format(fraction, only_train)

    if not os.path.isfile(cache_path):
        files = glob.glob(os.path.join(INPUT_PATH, 'train/*/*.jpg'))
        if not only_train:
              files += glob.glob(os.path.join(INPUT_PATH, 'external/*/*.jpg'))

        files = np.array(files)
        train, valid = train_test_split(files, train_size=fraction, random_state=66)
        save_in_file((train, valid), cache_path)
    else:
        train, valid = load_from_file(cache_path)

    # check all files exists
    if 1:
        files = list(train) + list(valid)
        print('Files in KFold split: {}'.format(len(files)))
        for f in files:
            if not os.path.isfile(f):
                print('File {} is absent!'.format(f))
                exit()

    return train, valid


def save_in_file(arr, file_name):
    pickle.dump(arr, gzip.open(file_name, 'wb+', compresslevel=3))


def load_from_file(file_name):
    return pickle.load(gzip.open(file_name, 'rb'))


def show_image(im, name='image'):
    cv2.imshow(name, im.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_resized_image(P, w=1000, h=1000):
    res = cv2.resize(P.astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    show_image(res)


def get_date_string():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")


def sort_dict_by_values(a, reverse=True):
    sorted_x = sorted(a.items(), key=operator.itemgetter(1), reverse=reverse)
    return sorted_x


def value_counts_for_list(lst):
    a = dict(Counter(lst))
    a = sort_dict_by_values(a, True)
    return a


def save_history_figure(history, path, columns=('acc', 'val_acc')):
    import matplotlib.pyplot as plt
    s = pd.DataFrame(history.history)
    plt.plot(s[list(columns)])
    plt.savefig(path)
    plt.close()


def get_class(class_name):
    global CLASSES
    if class_name in CLASSES:
        class_idx = CLASSES.index(class_name)
    elif class_name in EXTRA_CLASSES:
        class_idx = EXTRA_CLASSES.index(class_name)
    else:
        print(class_name)
        assert False
    assert class_idx in range(N_CLASSES)
    return class_idx

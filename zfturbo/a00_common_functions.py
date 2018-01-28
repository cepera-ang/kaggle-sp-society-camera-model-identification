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
from sklearn.model_selection import KFold
from collections import Counter, defaultdict
from multiprocessing import Process, Manager
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

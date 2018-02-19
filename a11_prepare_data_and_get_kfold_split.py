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
import cv2
from train import get_size_quality, get_crop, load_img_fast_jpg
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
INPUT_PATH = '/mnt/3E6EDD526EDD0395/'
output_path = '../input/external_all_small/'

OUTPUT_PATH = 'data/'

iphone_soft = [
    '10.0.2',
    '10.1.1',
    '10.2',
    '10.2.1',
    '10.3.1',
    '10.3.2',
    '10.3.3',
    '11.0.1',
    '11.0.2',
    '11.0.3',
    '11.1',
    '11.1.1',
    '11.1.2',
    '11.2',
    '11.2.1',
    '11.2.2',
    '5.0',
    '5.0.1',
    '5.1',
    '5.1.1',
    '6.0',
    '6.0.1',
    '6.1',
    '6.1.2',
    '6.1.3',
    '7.0',
    '7.0.2',
    '7.0.3',
    '7.0.4',
    '7.0.6',
    '7.1',
    '7.1.1',
    '7.1.2',
    '8.0',
    '8.0.2',
    '8.1',
    '8.1.1',
    '8.1.2',
    '8.1.3',
    '8.2',
    '8.3',
    '8.4',
    '8.4.1',
    '9.0',
    '9.0.1',
    '9.0.2',
    '9.1',
    '9.2',
    '9.2.1',
    '9.3',
    '9.3.1',
    '9.3.2',
    '9.3.3',
    '9.3.4',
    '9.3.5',
    '10.0',
    '11.0',
    'Microsoft Windows Photo Viewer 6.1.7600.16385',

]

good_software = [
  'HDR+',
  'bull',
  'I950',
  'N900',
  'NEX-',
  'none',
]

ALL_RESOLUTIONS = [
    [1520,2688], # flips
    [3264,2448], # no flips
    [2432,4320], # flips
    [3120,4160], # flips
    [4128,2322], # no flips
    [3264,2448], # no flips
    [3024,4032], # flips
    # [1040,780],  # Motorola-Nexus-6 no flips
    [3088,4130], [3120,4160], # Motorola-Nexus-6 flips
    [4128,2322], # no flips
    [6000,4000], # no flips
    [4160, 2340],
    [2340, 4160],
]
ALL_RESOLUTIONS.extend([resolution[::-1] for resolution in ALL_RESOLUTIONS])

exif_dict = {
    'HTC One': 'HTC-1-M7',
    'HTC6500LVW': 'HTC-1-M7',
    'HTCONE': 'HTC-1-M7',

    'Nexus 5X': 'LG-Nexus-5x',

    'XT1080': 'Motorola-Droid-Maxx',
    'XT1060': 'Motorola-Droid-Maxx',
    'XT1052': 'Motorola-Droid-Maxx',
    'XT1053': 'Motorola-Droid-Maxx',
    'XT1054': 'Motorola-Droid-Maxx',
    'XT1055': 'Motorola-Droid-Maxx',
    'XT1056': 'Motorola-Droid-Maxx',
    'XT1057': 'Motorola-Droid-Maxx',

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
    hash_checker = dict()
    files = glob.glob(raw_path + '**/*.jpg', recursive=True)
    print(raw_path)
    if os.path.isdir(output_path):
        print('Output folder "{}" already exists! You must delete it before proceed!'.format(output_path))
        y = input('Would you like to delete folder?')
        if y[:1] == 'y':
            shutil.rmtree(output_path)
        else:
            exit()
    os.mkdir(output_path)
    print('Files found: {}'.format(len(files)))

    p = Pool(1)
    # results = list of (hash, file)
    results = p.map(process_file, tqdm(files))
    for hsh, f in results:
        if (hsh is not None) and (hsh in hash_checker):
            print('Hash {} for file {} alread exists. Skip file!'.format(hsh, f))
            if f is not None:
                os.remove(f)
        hash_checker[hsh] = 1


    copied_files = glob.glob(output_path + '**/*.jpg', recursive=True)
    print('Files in external folder: {}'.format(len(copied_files)))
    return exif_dict

def process_file(f):
    try:
        tags = exifread.process_file(open(f, 'rb'))
        try:
            model = str(tags['Image Model'])
        except:
            print('Broken Image Model EXIF: {}'.format(f))
            return (None, None)
        if model not in exif_dict:
            print('Skip EXIF: Bad model {}'.format(model))
            return (None, None)
        try:
            software = str(tags['Image Software'])
        except:
            # print('Broken Image Software EXIF: {}'.format(f))
            software = 'none'
        if (software not in iphone_soft) and (software[:4] not in good_software) and (model[:2] != 'XT'):
            # print(software[:4])
            print('Skip EXIF: Bad soft  {}'.format(software[:50]))
            return (None, None)
        w, h, q = get_size_quality(f)
        shape = list((h, w))

        if (q < 94) or (shape not in ALL_RESOLUTIONS):
            print('Skip:', shape, q, f.split(sep='/')[-1])
            return (None, None)

        # Check unique hash
        hsh = md5_from_file(f)

        out_folder = output_path + exif_dict[model]
        if not os.path.isdir(out_folder):
            os.mkdir(out_folder)

        img = load_img_fast_jpg(f)
        shape = list((img.shape[1], img.shape[0]))
        if (shape not in ALL_RESOLUTIONS):
            print('Skip: shape ', shape, f.split(sep='/')[-1])
            return (None, None)
        if len(img.shape) != 3:
            print('Skip: no 3 dim', f.split(sep='/')[-1])
            return (None, None)

        img = get_crop(img, CROP_SIZE)
        file_name = f.split(sep='/')[-1][:-4] + '.tif'
        full_name = os.path.join(out_folder, file_name)
        # print('Saving file', file_name, os.path.join(out_folder, file_name))
        cv2.imwrite(full_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # shutil.copy2(f, out_folder)
        return (hsh, full_name)
    except:
        return (None, None)

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

CROP_SIZE = 1024
if __name__ == '__main__':
    # 1st param - location of your directories like 'flickr1', 'val_images' etc
    # 2nd parameter - location where files will be copied. Warning: you need to have sufficient space
    prepare_external_dataset(INPUT_PATH + 'external/', output_path)

    # will return list of lists [[train1, valid1], [train2, valid2] , ... [trainK, validK]]
    # kf = get_kfold_split(num_folds=4)

# Copyright team STAMP

if __name__ == '__main__':
    import os
    gpu_use = 2
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)

from a01_random_augmentations import *
import argparse
import glob
import numpy as np
import pandas as pd
import random
from os.path import isfile, join
from sklearn.utils import class_weight

from keras.optimizers import Adam, Adadelta, SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model, Model
from keras.layers import concatenate, Lambda, Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Reshape
from keras.utils import to_categorical
from keras.applications import *
from keras import backend as K

from PIL import Image
from io import BytesIO
import re
import os
from tqdm import tqdm
import jpeg4py as jpeg
import cv2
import math
import csv

from functools import partial
from itertools import  islice
from conditional import conditional

SEED = 42

np.random.seed(SEED)
random.seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('--max-epoch', type=int, default=200, help='Epoch to run')
parser.add_argument('-b', '--batch-size', type=int, default=16, help='Batch Size during training, e.g. -b 64')
parser.add_argument('-l', '--learning_rate', type=float, default=1e-4, help='Initial learning rate')
parser.add_argument('-m', '--model', help='load hdf5 model including weights (and continue training)')
parser.add_argument('-w', '--weights', help='load hdf5 weights only (and continue training)')
parser.add_argument('-do', '--dropout', type=float, default=0.3, help='Dropout rate for FC layers')
parser.add_argument('-doc', '--dropout-classifier', type=float, default=0., help='Dropout rate for classifier')
parser.add_argument('-t', '--test', action='store_true', help='Test model and generate CSV submission file')
parser.add_argument('-tt', '--test-train', action='store_true', help='Test model on the training set')
parser.add_argument('-cs', '--crop-size', type=int, default=512, help='Crop size')
parser.add_argument('-g', '--gpus', type=int, default=1, help='Number of GPUs to use')
parser.add_argument('-p', '--pooling', type=str, default='avg', help='Type of pooling to use: avg|max|none')
parser.add_argument('-nfc', '--no-fcs', action='store_true', help='Dont add any FC at the end, just a softmax')
parser.add_argument('-kf', '--kernel-filter', action='store_true', help='Apply kernel filter')
parser.add_argument('-lkf', '--learn-kernel-filter', action='store_true', help='Add a trainable kernel filter before classifier')
parser.add_argument('-cm', '--classifier', type=str, default='ResNet50', help='Base classifier model to use')
parser.add_argument('-uiw', '--use-imagenet-weights', action='store_true', help='Use imagenet weights (transfer learning)')
parser.add_argument('-x', '--extra-dataset', action='store_true', help='Use dataset from https://www.kaggle.com/c/sp-society-camera-model-identification/discussion/47235')
parser.add_argument('-v', '--verbose', action='store_true', help='Pring debug/verbose info')
parser.add_argument('-e', '--ensembling', type=str, default='arithmetic', help='Type of ensembling: arithmetic|geometric for TTA')
parser.add_argument('-tta', action='store_true', help='Enable test time augmentation')
parser.add_argument('--check-train', action='store_true', default=False, help='Enable checking of all train JPEGs to remove broken')

args = parser.parse_args()

for class_id,resolutions in RESOLUTIONS.copy().items():
    resolutions.extend([resolution[::-1] for resolution in resolutions])
    RESOLUTIONS[class_id] = resolutions


def print_distribution(ids, classes=None):
    if classes is None:
        classes = [get_class(idx.split('/')[-2]) for idx in ids]
    classes_count = np.bincount(classes)
    for class_name, class_count in zip(CLASSES, classes_count):
        print('{:>22}: {:5d} ({:04.1f}%)'.format(class_name, class_count, 100. * class_count / len(classes)))


def run_validation_single():
    print("Loading model " + args.model)
    model = load_model(args.model, compile=False)
    # e.g. ResNet50_do0.3_doc0.0_avg-fold_1-epoch053-val_acc0.911957.hdf5
    match = re.search(r'(([a-zA-Z0-9]+)_[A-Za-z_\d\.]+)-fold_(\d+)-epoch(\d+)-.*\.hdf5', args.model)
    args.classifier = match.group(2)
    fold_num = int(match.group(3))
    CROP_SIZE = args.crop_size  = model.get_input_shape_at(0)[0][1]
    print("Overriding classifier: {} and crop size: {} and fold num: {}".format(args.classifier, args.crop_size, fold_num))
    model.summary()

    # single_split = get_single_split(fraction=0.9)
    # single_split = get_single_split_with_csv_file(fraction=0.9, csv_file=OUTPUT_PATH + 'common_image_info_additional.csv')
    # kfold_split = get_kfold_split_with_csv_file(4, OUTPUT_PATH + 'common_image_info_additional.csv')
    single_split = get_single_split_final(OUTPUT_PATH + 'common_image_info_additional.csv',
                                          OUTPUT_PATH + 'validation_files.pklz')
    ids = list(single_split[1])
    print('Files to process: {}'.format(len(ids)))
    ids.sort()

    correct_predictions_no_manip = 0
    correct_predictions_with_manip = 0
    fnames = []
    real_class = []
    aug = []
    manip_arr = []
    probs = np.array([]*10).reshape((0, 10))

    manipulation_stat = dict()
    for m in MANIPULATIONS:
        manipulation_stat[m] = [0, 0]

    for i, idx in enumerate(ids):
        print('Go for {}'.format(idx))
        #fnames.append(idx.split("/")[-1])
        if 0:
            img_orig = np.array(Image.open(idx))
        else:
            img_orig = pyvips.Image.new_from_file(idx, access='sequential')
            img_orig = np.ndarray(buffer=img_orig.write_to_memory(),
                             dtype=np.uint8,
                             shape=[img_orig.height, img_orig.width, img_orig.bands])

        # Исходный вариант (берём 512 пикселов)
        img_init = get_crop(img_orig, 512, random_crop=False)
        j = 0
        img_batch = []
        manipulated_batch = []
        manipulated = 0.0

        for start0 in range(0, img_init.shape[0], 96):
            for start1 in range(0, img_init.shape[1], 96):
                end0 = start0 + CROP_SIZE
                if end0 > img_init.shape[0]:
                    continue
                end1 = start1 + CROP_SIZE
                if end1 > img_init.shape[1]:
                    continue
                img = img_init[start0:end0, start1:end1].copy()
                # print(start0, start1, img.shape)
                timg = cv2.transpose(img)
                for _img in [img, cv2.flip(img, 0), cv2.flip(img, 1), cv2.flip(img, -1),
                            timg, cv2.flip(timg, 0), cv2.flip(timg, 1), cv2.flip(timg, -1)]:
                    im_tmp = preprocess_image(_img, classifier=args.classifier)
                    img_batch.append(im_tmp)
                    manipulated_batch.append(manipulated)
                    fnames.append(idx.split("/")[-1])
                    aug.append(j)
                    manip_arr.append(manipulated)
                    j += 1
        img_batch = np.array(img_batch, dtype=np.float32)
        manipulated_batch = np.array(manipulated_batch, dtype=np.float32)

        l = img_batch.shape[0]
        # print('TTA size: {} J size: {}'.format(l, j))
        batch_size = args.batch_size
        for i in range(((l-1) // batch_size) + 1):
            # print(i*batch_size, min(l, (i+1)*batch_size))
            batch_pred = model.predict_on_batch([img_batch[i*batch_size: min(l, (i+1)*batch_size)],
                                                 manipulated_batch[i*batch_size: min(l, (i+1)*batch_size)]])
            if i == 0:
                prediction = batch_pred
            else:
                prediction = np.concatenate((prediction, batch_pred), axis=0)

        probs = np.vstack((probs, prediction))
        pred_no_manip = np.mean(prediction, axis=0)
        # prediction = np.sqrt(np.mean(prediction**2, axis=0))
        # prediction = scipy.stats.mstats.gmean(prediction, axis=0)

        prediction_class_idx = np.argmax(pred_no_manip)
        class_idx = get_class(os.path.basename(os.path.dirname(idx)))
        real_class.append(class_idx)
        correct_str = 'no'
        if class_idx == prediction_class_idx:
            correct_predictions_no_manip += 1
            correct_str = 'yes'

        print('Prediction no manip: {} Prob: {:.4f}'.format(np.argmax(pred_no_manip), pred_no_manip[np.argmax(pred_no_manip)]))
        print('Prediction real: {} [Correct: {}]'.format(class_idx, correct_str))

        # Все манипуляции
        for m in MANIPULATIONS:
            img_init = get_crop(img_orig, 2 * 512, random_crop=False)
            # m = random.choice(MANIPULATIONS)
            img_init = random_manipulation(img_init.copy(), m)
            img_init = get_crop(img_init, 512, random_crop=False)

            j = 0
            img_batch = []
            manipulated_batch = []
            manipulated = 1.0
            for start0 in range(0, img_init.shape[0], 96):
                for start1 in range(0, img_init.shape[1], 96):
                    end0 = start0 + CROP_SIZE
                    if end0 > img_init.shape[0]:
                        continue
                    end1 = start1 + CROP_SIZE
                    if end1 > img_init.shape[1]:
                        continue
                    img = img_init[start0:end0, start1:end1].copy()
                    timg = cv2.transpose(img)
                    for _img in [img, cv2.flip(img, 0), cv2.flip(img, 1), cv2.flip(img, -1),
                                 timg, cv2.flip(timg, 0), cv2.flip(timg, 1), cv2.flip(timg, -1)]:
                        img_tmp = preprocess_image(_img, classifier=args.classifier)
                        img_batch.append(img_tmp)
                        manipulated_batch.append(manipulated)
                        fnames.append(idx.split("/")[-1])
                        aug.append(j)
                        manip_arr.append(manipulated)
                        j += 1

            img_batch = np.array(img_batch, dtype=np.float32)
            manipulated_batch = np.array(manipulated_batch, dtype=np.float32)

            l = img_batch.shape[0]
            # print('TTA size: {} J size: {}'.format(l, j))
            batch_size = args.batch_size
            for i in range((l-1) // batch_size + 1):
                batch_pred = model.predict_on_batch([img_batch[i * batch_size:min(l, (i + 1) * batch_size)],
                                                     manipulated_batch[i * batch_size:min(l, (i + 1) * batch_size)]])
                if i == 0:
                    prediction = batch_pred
                else:
                    prediction = np.concatenate((prediction, batch_pred), axis=0)

            probs = np.vstack((probs, prediction))
            pred_with_manip = np.mean(prediction, axis=0)

            prediction_class_idx = np.argmax(pred_with_manip)
            class_idx = get_class(os.path.basename(os.path.dirname(idx)))
            real_class.append(class_idx)
            if class_idx == prediction_class_idx:
                correct_predictions_with_manip += 1
                correct_str = 'yes'
                manipulation_stat[m][0] += 1
            else:
                correct_str = 'no'
                manipulation_stat[m][1] += 1

            print('Prediction with manip {}: {} Prob: {:.4f}'.format(m, np.argmax(pred_with_manip), pred_with_manip[np.argmax(pred_with_manip)]))
            print('Prediction real: {} [Correct: {}]'.format(class_idx, correct_str))

    ans = pd.DataFrame()
    ans["name"] = fnames
    ans["aug"] = aug
    ans["manip"] = manip_arr
    for i in range(10):
        ans[CLASSES[i]] = probs[:, i]
    out_path = SUBM_PATH + "/tta_8_" + os.path.basename(args.model) + '_train.csv'
    pd.DataFrame(ans).to_csv(out_path, index=False)

    print('Manipulation stat')
    for m in MANIPULATIONS:
        print('{}: {}'.format(m, manipulation_stat[m]))

    correct_predictions_no_manip /= len(ids)
    correct_predictions_with_manip /= len(MANIPULATIONS) * len(ids)
    print("Accuracy no manipulation: {:.6f}".format(correct_predictions_no_manip))
    print("Accuracy with manipulation: {:.6f}".format(correct_predictions_with_manip))
    print("Accuracy overall: {:.6f}".format((0.7 * correct_predictions_no_manip + 0.3 * correct_predictions_with_manip)))


def check_subm_distribution(subm_path):
    df = pd.read_csv(subm_path)
    checker = dict()
    for c in CLASSES:
        checker[c] = [0, 0]

    manip = []
    for index, row in df.iterrows():
        if '_manip' in row['fname']:
            checker[row['camera']][0] += 1
            manip.append(1)
        else:
            checker[row['camera']][1] += 1
            manip.append(0)
    df['manip'] = manip

    for c in CLASSES:
        print('{}: {}'.format(c, checker[c]))


def check_subm_diff(s1p, s2p):
    df1 = pd.read_csv(s1p)
    df2 = pd.read_csv(s2p)
    df1.sort_values('fname', inplace=True)
    df1.reset_index(drop=True, inplace=True)
    df2.sort_values('fname', inplace=True)
    df2.reset_index(drop=True, inplace=True)
    dff = len(df1[df1['camera'] != df2['camera']])
    total = len(df1)
    perc = 100 * dff / total
    print('Difference in {} pos from {}. Percent: {:.2f}%'.format(dff, total, perc))


def proc_tst_and_create_subm():
    print("Loading model " + args.model)
    model = load_model(args.model, compile=False)
    # e.g. ResNet50_do0.3_doc0.0_avg-fold_1-epoch053-val_acc0.911957.hdf5
    match = re.search(r'(([a-zA-Z0-9]+)_[A-Za-z_\d\.]+)-fold_(\d+)-epoch(\d+)-.*\.hdf5', args.model)
    args.classifier = match.group(2)
    fold_num = int(match.group(3))
    CROP_SIZE = args.crop_size  = model.get_input_shape_at(0)[0][1]
    print("Overriding classifier: {} and crop size: {} and fold num: {}".format(args.classifier, args.crop_size, fold_num))
    model.summary()

    ids = glob.glob(join(INPUT_PATH, 'test/*.tif'))
    print('Files to process: {}'.format(len(ids)))
    ids.sort()

    submission_file = SUBM_PATH + 'submission_{}.csv'.format(args.model.split(sep='/')[-1])
    csvfile = open(submission_file, 'w')
    csv_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['fname', 'camera'])
    classes = []

    fnames = []
    aug = []
    manip_arr = []
    probs = np.array([]*10).reshape((0, 10))
    for i, idx in enumerate(tqdm(ids)):
        img_init = np.array(Image.open(idx))
        manipulated = 1. if idx.find('manip') != -1 else 0.
        j = 0
        img_batch = []
        manipulated_batch = []
        for start0 in range(0, img_init.shape[0], 96):
            for start1 in range(0, img_init.shape[1], 96):
                end0 = start0 + CROP_SIZE
                if end0 > img_init.shape[0]:
                    continue
                end1 = start1 + CROP_SIZE
                if end1 > img_init.shape[1]:
                    continue
                img = img_init[start0:end0, start1:end1].copy()
                timg = cv2.transpose(img)
                for _img in [img, cv2.flip(img, 0), cv2.flip(img, 1), cv2.flip(img, -1),
                            timg, cv2.flip(timg, 0), cv2.flip(timg, 1), cv2.flip(timg, -1)]:
                    img_tmp = preprocess_image(_img, classifier=args.classifier)
                    img_batch.append(img_tmp)
                    manipulated_batch.append(manipulated)
                    fnames.append(idx.split("/")[-1])
                    aug.append(j)
                    manip_arr.append(manipulated)
                    j += 1

        img_batch = np.array(img_batch, dtype=np.float32)
        manipulated_batch = np.array(manipulated_batch, dtype=np.float32)

        l = img_batch.shape[0]
        batch_size = args.batch_size
        for i in range(((l-1) // batch_size) + 1):
            batch_pred = model.predict_on_batch([img_batch[i*batch_size:min(l,(i+1)*batch_size)],
                                                 manipulated_batch[i*batch_size:min(l,(i+1)*batch_size)]])
            if i == 0:
                prediction = batch_pred
            else:
                prediction = np.concatenate((prediction, batch_pred), axis=0)

        probs = np.vstack((probs, prediction))
        if prediction.shape[0] != 1: # TTA
            prediction = np.mean(prediction, axis=0)
            # prediction = np.sqrt(np.mean(prediction**2, axis=0))
            # prediction = scipy.stats.mstats.gmean(prediction, axis=0)

        prediction_class_idx = np.argmax(prediction)
        csv_writer.writerow([os.path.basename(idx), CLASSES[prediction_class_idx]])
        classes.append(prediction_class_idx)

    ans = pd.DataFrame()
    ans["name"] = fnames
    ans["aug"] = aug
    ans["manip"] = manip_arr
    for i in range(10):
        ans[CLASSES[i]] = probs[:, i]
    out_path = SUBM_PATH + "/tta_8_" + os.path.basename(args.model) + '_test.csv'
    pd.DataFrame(ans).to_csv(out_path, index=False)

    print("Test set predictions distribution:")
    print_distribution(None, classes=classes)

    csvfile.close()
    check_subm_distribution(submission_file)
    check_subm_diff(SUBM_PATH + '0.991_equal_2_pwr_mean_hun_5_prod-ce..csv', submission_file)


if __name__ == '__main__':
    start_time = time.time()
    args.model = MODELS_PATH + 'DenseNet169_do0.3_doc0.0_avg-fold_1-epoch080-val_acc0.970985.hdf5'

    if 1:
        # Validation
        args.test_train = True
        args.test = None
        run_validation_single()

    if 1:
        # Test
        args.test_train = None
        args.test = True
        proc_tst_and_create_subm()

    print('Time: {:.0f} sec'.format(time.time() - start_time))

'''
DenseNet169_do0.3_doc0.0_avg-fold_1-epoch080-val_acc0.970985.hdf5
HTC-1-M7: [132, 132]
iPhone-6: [130, 131]
Motorola-Droid-Maxx: [132, 133]
Motorola-X: [132, 132]
Samsung-Galaxy-S4: [134, 133]
iPhone-4s: [134, 132]
LG-Nexus-5x: [75, 109]
Motorola-Nexus-6: [151, 140]
Samsung-Galaxy-Note3: [160, 145]
Sony-NEX-7: [140, 133]
Difference in 101 pos from 2640. Percent: 3.83%

'''
# Copyright team STAMP

if __name__ == '__main__':
    import os
    gpu_use = 1
    FOLD_TO_CALC = [gpu_use+1]
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
    from conditional import conditional

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
    kfold_split = get_kfold_split_with_csv_file(4, OUTPUT_PATH + 'common_image_info_additional.csv')
    if args.test:
        ids = glob.glob(join(INPUT_PATH, 'test/*.tif'))
    elif args.test_train:
        # ids = single_split[1]
        ids = kfold_split[fold_num-1][1]
        # ids = ids
    else:
        assert False

    print('Files to process: {}'.format(len(ids)))
    ids.sort()

    correct_predictions = 0
    submission_file = 'submission_{}.csv'.format(args.model.split(sep='/')[-1])
    with conditional(args.test, open(join(SUBM_PATH, submission_file), 'w')) as csvfile:
        if args.test:
            csv_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['fname', 'camera'])
            classes = []

        fnames = []
        real_class = []
        labels = []
        aug = []
        probs = np.array([]*10).reshape((0,10))
        for i, idx in enumerate(tqdm(ids)):
            #fnames.append(idx.split("/")[-1])
            if 0:
                img = np.array(Image.open(idx))
            else:
                img = pyvips.Image.new_from_file(idx, access='sequential')
                img = np.ndarray(buffer=img.write_to_memory(),
                                 dtype=np.uint8,
                                 shape=[img.height, img.width, img.bands])

            if args.test_train:
                img = get_crop(img, CROP_SIZE, random_crop=False)

            manipulated = np.float32([1. if idx.find('manip') != -1 else 0.])

            sx = img.shape[1] // CROP_SIZE
            sy = img.shape[0] // CROP_SIZE
            j = 0
            k = 8
            img_batch = np.zeros((k * sx * sy, CROP_SIZE, CROP_SIZE, 3), dtype=np.float32)
            manipulated_batch = np.zeros((k * sx * sy, 1),  dtype=np.float32)
            timg = cv2.transpose(img)
            for _img in [img, cv2.flip(img, 0), cv2.flip(img, 1), cv2.flip(img, -1),
                        timg, cv2.flip(timg, 0), cv2.flip(timg, 1), cv2.flip(timg, -1)]:
                img_batch[j]         = preprocess_image(_img, classifier=args.classifier)
                manipulated_batch[j] = manipulated
                fnames.append(idx.split("/")[-1])
                aug.append(j)
                j += 1

            l = img_batch.shape[0]
            batch_size = args.batch_size
            for i in range(l//batch_size+1):
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

            if args.test_train:
                class_idx = get_class(os.path.basename(os.path.dirname(idx)))
                real_class.append(class_idx)
                if class_idx == prediction_class_idx:
                    correct_predictions += 1

            if args.test:
                csv_writer.writerow([os.path.basename(idx), CLASSES[prediction_class_idx]])
                classes.append(prediction_class_idx)

        ans = pd.DataFrame()
        ans["name"] = fnames
        ans["aug"] = aug
        for i in range(10):
            ans[CLASSES[i]] = probs[:, i]
        if args.test_train:
            out_path = SUBM_PATH + "/tta_8_" + os.path.basename(args.model) + '_train.csv'
        else:
            out_path = SUBM_PATH + "/tta_8_" + os.path.basename(args.model) + '_test.csv'
        pd.DataFrame(ans).to_csv(out_path, index=False)

        if args.test_train:
            print("Accuracy: " + str(correct_predictions / len(ids)))

        if args.test:
            print("Test set predictions distribution:")
            print_distribution(None, classes=classes)


if __name__ == '__main__':
    start_time = time.time()
    args.model = MODELS_PATH + 'VGG16_do0.3_doc0.0_avg-fold_1-epoch042-val_acc0.887476.hdf5'

    # Validation
    args.test_train = True
    args.test = None
    run_validation_single()

    # Test
    args.test_train = None
    args.test = True
    run_validation_single()

    print('Time: {:.0f} sec'.format(time.time() - start_time))

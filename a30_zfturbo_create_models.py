# Copyright team STAMP

from a00_common_functions import *
import argparse
import glob
import numpy as np
import random
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from keras.optimizers import Adam, Adadelta, SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model, Model
from keras.layers import concatenate, Lambda, Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, \
        BatchNormalization, Activation, GlobalAveragePooling2D, Reshape
from multi_gpu_keras import multi_gpu_model

from PIL import Image
from io import BytesIO
import re
import os
import jpeg4py as jpeg
import cv2
import math
from multiprocessing import Pool, cpu_count
from functools import partial
from itertools import  islice


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

args = parser.parse_args()

TRAIN_FOLDER = '../input/train'
EXTRA_TRAIN_FOLDER = '../input/flickr_images'
NEW_TRAIN_FOLDER = '../input/flickr_new'
TEST_FOLDER  = '../input/test'
EXTRA_VAL_FOLDER = '../input/val_images'
MODEL_FOLDER = '../models/'

CROP_SIZE = args.crop_size

for class_id, resolutions in RESOLUTIONS.copy().items():
    resolutions.extend([resolution[::-1] for resolution in resolutions])
    RESOLUTIONS[class_id] = resolutions

MANIPULATIONS = ['jpg70', 'jpg90', 'gamma0.8', 'gamma1.2', 'bicubic0.5', 'bicubic0.8', 'bicubic1.5', 'bicubic2.0']
VALIDATION_TRANSFORMS = [[], ['orientation'], ['manipulation'], ['orientation', 'manipulation']]

N_CLASSES = len(CLASSES)
load_img_fast_jpg  = lambda img_path: jpeg.JPEG(img_path).decode()
load_img           = lambda img_path: np.array(Image.open(img_path))


def check_remove_broken(img_path):
    try:
        x = jpeg.JPEG(img_path).decode()
    except Exception:
        print('Decoding error:', img_path)
        os.remove(img_path)


def random_manipulation(img, manipulation=None):

    if manipulation is None:
        manipulation = random.choice(MANIPULATIONS)

    if manipulation.startswith('jpg'):
        quality = int(manipulation[3:])
        out = BytesIO()
        im = Image.fromarray(img)
        im.save(out, format='jpeg', quality=quality)
        im_decoded = jpeg.JPEG(np.frombuffer(out.getvalue(), dtype=np.uint8)).decode()
        del out
        del im
    elif manipulation.startswith('gamma'):
        gamma = float(manipulation[5:])
        # alternatively use skimage.exposure.adjust_gamma
        # img = skimage.exposure.adjust_gamma(img, gamma)
        im_decoded = np.uint8(cv2.pow(img / 255., gamma)*255.)
    elif manipulation.startswith('bicubic'):
        scale = float(manipulation[7:])
        im_decoded = cv2.resize(img,(0,0), fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
    else:
        assert False
    return im_decoded


def preprocess_image(img):
    
    if args.kernel_filter:
        # see slide 13
        # http://www.lirmm.fr/~chaumont/publications/WIFS-2016_TUAMA_COMBY_CHAUMONT_Camera_Model_Identification_With_CNN_slides.pdf
        kernel_filter = 1/12. * np.array([\
            [-1,  2,  -2,  2, -1],  \
            [ 2, -6,   8, -6,  2],  \
            [-2,  8, -12,  8, -2],  \
            [ 2, -6,   8, -6,  2],  \
            [-1,  2,  -2,  2, -1]]) 

        return cv2.filter2D(img.astype(np.float32),-1,kernel_filter)
        # kernel filter already puts mean ~0 and roughly scales between [-1..1]
        # no need to preprocess_input further
    else:
        # find `preprocess_input` function specific to the classifier
        classifier_to_module = { 
            'NASNetLarge'       : 'nasnet',
            'NASNetMobile'      : 'nasnet',
            'DenseNet40'        : 'densenet',
            'DenseNet121'       : 'densenet',
            'DenseNet161'       : 'densenet',
            'DenseNet201'       : 'densenet',
            'InceptionResNetV2' : 'inception_resnet_v2',
            'InceptionV3'       : 'inception_v3',
            'MobileNet'         : 'mobilenet',
            'ResNet50'          : 'resnet50',
            'VGG16'             : 'vgg16',
            'VGG19'             : 'vgg19',
            'Xception'          : 'xception',

        }

        if args.classifier in classifier_to_module:
            classifier_module_name = classifier_to_module[args.classifier]
        else:
            classifier_module_name = 'xception'

        preprocess_input_function = getattr(globals()[classifier_module_name], 'preprocess_input')
        return preprocess_input_function(img.astype(np.float32))


def get_crop(img, crop_size, random_crop=False):
    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    half_crop = crop_size // 2
    pad_x = max(0, crop_size - img.shape[1])
    pad_y = max(0, crop_size - img.shape[0])
    if (pad_x > 0) or (pad_y > 0):
        img = np.pad(img, ((pad_y//2, pad_y - pad_y//2), (pad_x//2, pad_x - pad_x//2), (0,0)), mode='wrap')
        center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    if random_crop:
        freedom_x, freedom_y = img.shape[1] - crop_size, img.shape[0] - crop_size
        if freedom_x > 0:
            center_x += np.random.randint(math.ceil(-freedom_x/2), freedom_x - math.floor(freedom_x/2) )
        if freedom_y > 0:
            center_y += np.random.randint(math.ceil(-freedom_y/2), freedom_y - math.floor(freedom_y/2) )

    return img[center_y - half_crop : center_y + crop_size - half_crop, center_x - half_crop : center_x + crop_size - half_crop]


def get_class(class_name):
    if class_name in CLASSES:
        class_idx = CLASSES.index(class_name)
    elif class_name in EXTRA_CLASSES:
        class_idx = EXTRA_CLASSES.index(class_name)
    else:
        assert False
    assert class_idx in range(N_CLASSES)
    return class_idx


def process_item(item, training, transforms=[[]]):

    class_name = item.split('/')[-2]
    class_idx = get_class(class_name)

    validation = not training 

    img = load_img_fast_jpg(item)

    shape = list(img.shape[:2])

    # discard images that do not have right resolution
    if shape not in RESOLUTIONS[class_idx]:
        return None

    # some images may not be downloaded correctly and are B/W, discard those
    if img.ndim != 3:
        return None

    if len(transforms) == 1:
        _img = img
    else:
        _img = np.copy(img)

        img_s         = [ ]
        manipulated_s = [ ]
        class_idx_s   = [ ]

    for transform in transforms:

        force_manipulation = 'manipulation' in transform

        if ('orientation' in transform) and (ORIENTATION_FLIP_ALLOWED[class_idx] is False):
            continue

        force_orientation  = ('orientation'  in transform) and ORIENTATION_FLIP_ALLOWED[class_idx]

        # some images are landscape, others are portrait, so augment training by randomly changing orientation
        if ((np.random.rand() < 0.5) and training and ORIENTATION_FLIP_ALLOWED[class_idx]) or force_orientation:
            img = np.rot90(_img, 1, (0,1))
            # is it rot90(..3..), rot90(..1..) or both? 
            # for phones with landscape mode pics could be taken upside down too, although less likely
            # most of the test images that are flipped are 1
            # however,eg. img_4d7be4c_unalt looks 3
            # and img_4df3673_manip img_6a31fd7_unalt looks 2!
        else:
            img = _img

        img = get_crop(img, CROP_SIZE * 2, random_crop=True if training else False) 
        # * 2 bc may need to scale by 0.5x and still get a 512px crop

        if args.verbose:
            print("om: ", img.shape, item)

        manipulated = 0.
        if ((np.random.rand() < 0.5) and training) or force_manipulation:
            img = random_manipulation(img)
            manipulated = 1.
            if args.verbose:
                print("am: ", img.shape, item)

        img = get_crop(img, CROP_SIZE, random_crop=True if training else False)
        if args.verbose:
            print("ac: ", img.shape, item)

        img = preprocess_image(img)
        if args.verbose:
            print("ap: ", img.shape, item)

        if len(transforms) > 1:
            img_s.append(img)    
            manipulated_s.append(manipulated)
            class_idx_s.append(class_idx)

    if len(transforms) == 1:
        return img, manipulated, class_idx
    else:
        return img_s, manipulated_s, class_idx_s


def gen(items, batch_size, training=True):

    validation = not training 

    # during validation we store the unaltered images on batch_idx and a manip one on batch_idx + batch_size, hence the 2
    valid_batch_factor = 1 # TODO: augment validation

    # X holds image crops
    X = np.empty((batch_size * valid_batch_factor, CROP_SIZE, CROP_SIZE, 3), dtype=np.float32)
    # O whether the image has been manipulated (1.) or not (0.)
    O = np.empty((batch_size * valid_batch_factor, 1), dtype=np.float32)

    # class index
    y = np.empty((batch_size * valid_batch_factor), dtype=np.int64)
    
    p = Pool(cpu_count()-2)

    transforms = VALIDATION_TRANSFORMS if validation else [[]]

    while True:

        if training:
            random.shuffle(items)

        process_item_func = partial(process_item, training=training, transforms=transforms)

        batch_idx = 0
        iter_items = iter(items)
        for item_batch in iter(lambda:list(islice(iter_items, batch_size)), []):

            batch_results = p.map(process_item_func, item_batch)
            for batch_result in batch_results:

                if batch_result is not None:
                    if len(transforms) == 1:
                        X[batch_idx], O[batch_idx], y[batch_idx] = batch_result
                        batch_idx += 1
                    else:
                        for _X,_O,_y in zip(*batch_result):
                            X[batch_idx], O[batch_idx], y[batch_idx] = _X,_O,_y
                            batch_idx += 1
                            if batch_idx == batch_size:
                                yield([X, O], [y])
                                batch_idx = 0

                if batch_idx == batch_size:
                    yield([X, O], [y])
                    batch_idx = 0


def print_distribution(ids, classes=None):
    if classes is None:
        classes = [get_class(idx.split('/')[-2]) for idx in ids]
    classes_count = np.bincount(classes)
    for class_name, class_count in zip(CLASSES, classes_count):
        print('{:>22}: {:5d} ({:04.1f}%)'.format(class_name, class_count, 100. * class_count / len(classes)))


def create_models(nfolds):
    global model, CROP_SIZE

    # MAIN
    if args.model:
        print("Loading model " + args.model)

        model = load_model(args.model, compile=False)
        # e.g. DenseNet201_do0.3_doc0.0_avg-epoch128-val_acc0.964744.hdf5
        match = re.search(r'(([a-zA-Z0-9]+)_[A-Za-z_\d\.]+)-epoch(\d+)-.*\.hdf5', args.model)
        model_name = match.group(1)
        args.classifier = match.group(2)
        CROP_SIZE = args.crop_size = model.get_input_shape_at(0)[0][1]
        print("Overriding classifier: {} and crop size: {}".format(args.classifier, args.crop_size))
        last_epoch = int(match.group(3))
    else:
        last_epoch = 0

        input_image = Input(shape=(CROP_SIZE, CROP_SIZE, 3))
        manipulated = Input(shape=(1,))

        classifier = globals()[args.classifier]

        classifier_model = classifier(
            include_top=False,
            weights='imagenet' if args.use_imagenet_weights else None,
            input_shape=(CROP_SIZE, CROP_SIZE, 3),
            pooling=args.pooling if args.pooling != 'none' else None)

        x = input_image
        if args.learn_kernel_filter:
            x = Conv2D(3, (7, 7), strides=(1, 1), use_bias=False, padding='valid', name='filtering')(x)
        x = classifier_model(x)
        x = Reshape((-1,))(x)
        if args.dropout_classifier != 0.:
            x = Dropout(args.dropout_classifier, name='dropout_classifier')(x)
        x = concatenate([x, manipulated])
        if not args.no_fcs:
            x = Dense(512, activation='relu', name='fc1')(x)
            x = Dropout(args.dropout, name='dropout_fc1')(x)
            x = Dense(128, activation='relu', name='fc2')(x)
            x = Dropout(args.dropout, name='dropout_fc2')(x)
        prediction = Dense(N_CLASSES, activation="softmax", name="predictions")(x)

        model = Model(inputs=(input_image, manipulated), outputs=prediction)
        model_name = args.classifier + \
                     ('_kf' if args.kernel_filter else '') + \
                     ('_lkf' if args.learn_kernel_filter else '') + \
                     '_do' + str(args.dropout) + \
                     '_doc' + str(args.dropout_classifier) + \
                     '_' + args.pooling

        if args.weights:
            model.load_weights(args.weights, by_name=True, skip_mismatch=True)
            match = re.search(r'([A-Za-z_\d\.]+)-epoch(\d+)-.*\.hdf5', args.weights)
            last_epoch = int(match.group(2))

    model.summary()
    model = multi_gpu_model(model, gpus=args.gpus)

    # TRAINING
    ids = glob.glob(join(TRAIN_FOLDER, '*/*.jpg'))
    ids.sort()

    if not args.extra_dataset:
        ids_train, ids_val = train_test_split(ids, test_size=0.1, random_state=SEED)
    else:
        ids_train = ids
        ids_val   = [ ]

        extra_train_ids = [os.path.join(EXTRA_TRAIN_FOLDER,line.rstrip('\n')) \
            for line in open(os.path.join(EXTRA_TRAIN_FOLDER, 'good_jpgs'))]
        low_quality =     [os.path.join(EXTRA_TRAIN_FOLDER,line.rstrip('\n').split(' ')[0]) \
            for line in open(os.path.join(EXTRA_TRAIN_FOLDER, 'low-quality'))]
        extra_train_ids = [idx for idx in extra_train_ids if idx not in low_quality]
        #extra_train_ids = [c for c in extra_train_ids if isfile(c)]
        extra_train_ids.sort()
        ids_train.extend(extra_train_ids)
        random.shuffle(ids_train)

        extra_val_ids = glob.glob(join(EXTRA_VAL_FOLDER,'*/*.jpg'))
        extra_val_ids.sort()
        ids_val.extend(extra_val_ids)

        classes_val = [get_class(idx.split('/')[-2]) for idx in ids_val]
        classes_val_count = np.bincount(classes_val)
        max_classes_val_count = max(classes_val_count)

        # Balance validation dataset by filling up classes with less items from training set (and removing those from there)
        for class_idx in range(N_CLASSES):
            idx_to_transfer = [idx for idx in ids_train \
                if get_class(idx.split('/')[-2]) == class_idx][:max_classes_val_count-classes_val_count[class_idx]]

            ids_train = list(set(ids_train).difference(set(idx_to_transfer)))

            ids_val.extend(idx_to_transfer)
        
        new_train_ids = glob.glob(join(NEW_TRAIN_FOLDER,'*/*.jpg'))
        #p = Pool(cpu_count() - 2)
        #p.map(check_remove_broken, tqdm(new_train_ids))
        #new_train_ids = glob.glob(join(NEW_TRAIN_FOLDER,'*/*.jpg'))
        ids_train.extend(new_train_ids)
        
        random.shuffle(ids_train)
        random.shuffle(ids_val)

    print("Training set distribution:")
    print_distribution(ids_train)

    print("Validation set distribution:")
    print_distribution(ids_val)

    classes_train = [get_class(idx.split('/')[-2]) for idx in ids_train]
    class_weight1 = class_weight.compute_class_weight('balanced', np.unique(classes_train), classes_train)

    opt = Adam(lr=args.learning_rate)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    metric  = "-val_acc{val_acc:.6f}"
    monitor = 'val_acc'

    save_checkpoint = ModelCheckpoint(
            join(MODEL_FOLDER, model_name+"-epoch{epoch:03d}"+metric+".hdf5"),
            monitor = monitor,
            verbose=0,  save_best_only=True, save_weights_only=False, mode='max', period=1)
    reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=5, min_lr=1e-9, epsilon = 0.00001, verbose=1, mode='max')

    model.fit_generator(
            generator        = gen(ids_train, args.batch_size),
            steps_per_epoch  = int(math.ceil(len(ids_train)  // args.batch_size)),
            validation_data  = gen(ids_val, args.batch_size, training=False),
            validation_steps = int(len(VALIDATION_TRANSFORMS) * math.ceil(len(ids_val) // args.batch_size)),
            epochs = args.max_epoch,
            callbacks = [save_checkpoint, reduce_lr],
            initial_epoch = last_epoch,
            max_queue_size = 10,
            class_weight=class_weight1)


if __name__ == '__main__':
    start_time = time.time()
    create_models(4)
    print('Time: {:.0f} sec'.format(time.time() - start_time))

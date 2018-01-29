# Copyright team STAMP

if __name__ == '__main__':
    import os
    gpu_use = 2
    FOLD_TO_CALC = [1, 2, 3, 4]
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


from a01_random_augmentations import *
from keras.applications import *
import argparse
import numpy as np
import random
from os.path import isfile, join
from sklearn.utils import class_weight

from PIL import Image
import re
import os
import cv2
import math
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from functools import partial
from itertools import islice


parser = argparse.ArgumentParser()
parser.add_argument('--max-epoch', type=int, default=200, help='Epoch to run')
parser.add_argument('-b', '--batch-size', type=int, default=10, help='Batch Size during training, e.g. -b 64')
parser.add_argument('-l', '--learning_rate', type=float, default=1e-5, help='Initial learning rate')
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
CROP_SIZE = args.crop_size

for class_id, resolutions in RESOLUTIONS.copy().items():
    resolutions.extend([resolution[::-1] for resolution in resolutions])
    RESOLUTIONS[class_id] = resolutions


VALIDATION_TRANSFORMS = [[], ['orientation'], ['manipulation'], ['orientation', 'manipulation']]
load_img           = lambda img_path: np.array(Image.open(img_path))


def gen(items, batch_size, training=True):

    validation = not training 

    # during validation we store the unaltered images on batch_idx and a manip one on batch_idx + batch_size, hence the 2
    valid_batch_factor = 1

    # X holds image crops
    X = np.empty((batch_size * valid_batch_factor, CROP_SIZE, CROP_SIZE, 3), dtype=np.float32)
    # O whether the image has been manipulated (1.) or not (0.)
    O = np.empty((batch_size * valid_batch_factor, 1), dtype=np.float32)

    # class index
    y = np.empty((batch_size * valid_batch_factor), dtype=np.int64)

    if 1:
        # p = Pool(cpu_count()-2)
        p = ThreadPool(cpu_count()-2)

    transforms = VALIDATION_TRANSFORMS if validation else [[]]

    while True:

        if training:
            random.shuffle(items)

        if 1:
            process_item_func = partial(process_item, training=training, transforms=transforms, crop_size=CROP_SIZE, classifier=args.classifier)

        batch_idx = 0
        iter_items = iter(items)
        for item_batch in iter(lambda:list(islice(iter_items, batch_size)), []):

            if 1:
                batch_results = p.map(process_item_func, item_batch)
            else:
                batch_results = []
                for it in item_batch:
                    b = process_item(it, training=training, transforms=transforms, crop_size=CROP_SIZE, classifier=args.classifier)
                    batch_results.append(b)

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
        classes = [get_class(os.path.basename(os.path.dirname(idx))) for idx in ids]
    classes_count = np.bincount(classes)
    for class_name, class_count in zip(CLASSES, classes_count):
        print('{:>22}: {:5d} ({:04.1f}%)'.format(class_name, class_count, 100. * class_count / len(classes)))


def create_models(nfolds):
    # from keras.applications import ResNet50

    from keras.optimizers import Adam, Adadelta, SGD
    from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
    from keras.models import load_model, Model
    from keras.layers import concatenate, Lambda, Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, \
        BatchNormalization, Activation, GlobalAveragePooling2D, Reshape
    from multi_gpu_keras import multi_gpu_model


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

        # classifier = globals()[args.classifier]

        classifier_model = ResNet50(
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
    # model = multi_gpu_model(model, gpus=args.gpus)

    # TRAINING
    num_fold = 0
    kfold_split = get_kfold_split(nfolds)
    for ids_train, ids_val in kfold_split:
        num_fold += 1
        print('Train files: {}'.format(len(ids_train)))
        print('Valid files: {}'.format(len(ids_val)))

        if 'FOLD_TO_CALC' in globals():
            if num_fold not in FOLD_TO_CALC:
                continue

        ids_train = list(ids_train)
        ids_val = list(ids_val)

        random.shuffle(ids_train)
        random.shuffle(ids_val)

        print("Training set distribution:")
        print_distribution(ids_train)

        print("Validation set distribution:")
        print_distribution(ids_val)

        classes_train = [get_class(os.path.basename(os.path.dirname(idx))) for idx in ids_train]
        class_weight1 = class_weight.compute_class_weight('balanced', np.unique(classes_train), classes_train)

        opt = Adam(lr=args.learning_rate)
        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        metric  = "-val_acc{val_acc:.6f}"
        monitor = 'val_acc'

        final_model_path = MODELS_PATH + '{}_fold_{}.h5'.format(args.classifier, num_fold)
        cache_model_path = MODELS_PATH + '{}_temp_fold_{}.h5'.format(args.classifier, num_fold)

        save_checkpoint2 = ModelCheckpoint(cache_model_path, monitor=monitor, save_best_only=True, verbose=0)
        save_checkpoint = ModelCheckpoint(
                join(MODELS_PATH, model_name + "-fold_{}".format(num_fold) + "-epoch{epoch:03d}" + metric + ".hdf5"),
                monitor=monitor,
                verbose=0, save_best_only=True, save_weights_only=False, mode='max', period=1)
        reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=5, min_lr=1e-9, epsilon=0.00001, verbose=1, mode='max')

        history = model.fit_generator(
                generator        = gen(ids_train, args.batch_size),
                steps_per_epoch  = int(math.ceil(len(ids_train)  // args.batch_size)),
                validation_data  = gen(ids_val, args.batch_size, training=False),
                validation_steps = int(len(VALIDATION_TRANSFORMS) * math.ceil(len(ids_val) // args.batch_size)),
                epochs=args.max_epoch,
                callbacks=[save_checkpoint, save_checkpoint2, reduce_lr],
                initial_epoch=last_epoch,
                max_queue_size=20,
                use_multiprocessing=False,
                workers=1,
                class_weight=class_weight1)

        max_acc = max(history.history[monitor])
        print('Maximum acc for fold {}: {} [Ep: {}]'.format(num_fold, max_acc, len(history.history[monitor])))
        model.load_weights(cache_model_path)
        model.save(final_model_path)
        now = datetime.datetime.now()
        filename = HISTORY_FOLDER_PATH + 'history_{}_{}_{:.4f}_lr_{}_{}.csv'.format(args.classifier, num_fold, max_acc,
                                                                                    args.learning_rate,
                                                                                    now.strftime("%Y-%m-%d-%H-%M"))
        pd.DataFrame(history.history).to_csv(filename, index=False)
        save_history_figure(history, filename[:-4] + '.png')


if __name__ == '__main__':
    start_time = time.time()
    create_models(4)
    print('Time: {:.0f} sec'.format(time.time() - start_time))

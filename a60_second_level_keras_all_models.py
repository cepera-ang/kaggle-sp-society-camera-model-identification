# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

'''
Second level model, which uses all previously generated features, based on Keras classifier
'''

if __name__ == '__main__':
    import os
    gpu_use = 0
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


import math
import datetime
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight
from a00_common_functions import *
from a60_second_level_xgboost_all_models import read_tables, rename_columns, check_subm_distribution, get_kfold_split_xgboost

random.seed(gpu_use)

def batch_generator_train_blender_random_sample(X, y, batch_size):
    rng = list(range(X.shape[0]))

    while True:
        index1 = random.sample(rng, batch_size)
        input1 = X[index1, :]
        output1 = y[index1]
        yield input1, output1


def ZF_random_keras_blender(input_features):
    from keras.models import Model
    from keras.layers import Input, Dense
    from keras.layers.core import Dropout

    layers_number = random.randint(1, 2)
    layer1_neurons = random.randint(input_features // 2, 2*input_features)
    layer2_neurons = random.randint(input_features // 2, layer1_neurons)
    layer1_droupout = random.uniform(0.3, 0.6)
    layer2_droupout = random.uniform(0.3, 0.6)
    layer1_activation = random.choice(['relu', 'sigmoid', 'tanh'])
    layer2_activation = random.choice(['relu', 'sigmoid', 'tanh'])

    inputs1 = Input((input_features,))
    x = Dense(layer1_neurons, activation=layer1_activation)(inputs1)
    x = Dropout(layer1_droupout)(x)
    if layers_number == 2:
        x = Dense(layer2_neurons, activation=layer2_activation)(x)
        x = Dropout(layer2_droupout)(x)
    x = Dense(10, activation='sigmoid', name='predictions')(x)

    model = Model(inputs=inputs1, outputs=x)
    print(model.summary())
    return model


def ZF_keras_blender_v3(input_features):
    from keras.models import Model
    from keras.layers import Input, Dense
    from keras.layers.core import Dropout

    inputs1 = Input((input_features,))
    x = Dense(input_features, activation='relu')(inputs1)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='sigmoid', name='predictions')(x)

    model = Model(inputs=inputs1, outputs=x)
    print(model.summary())
    return model


def create_keras_blender_model(train, features, num_iters):
    from keras import __version__
    import keras.backend as K
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras.optimizers import Adam, SGD
    from keras.utils import to_categorical

    print('Keras version: {}'.format(__version__))
    start_time = time.time()

    rescaled = len(train)
    model_list = []
    full_preds = np.zeros((rescaled, len(CLASSES)), dtype=np.float32)
    counts = np.zeros((rescaled, len(CLASSES)), dtype=np.float32)

    for iter in range(num_iters):
        num_folds = random.randint(3, 5)
        print('Iteration: {} Train shape: {}'.format(iter, train.shape))
        ret = get_kfold_split_xgboost(train, num_folds, iter + round(time.time()) % 10000)

        fold_num = 0
        for train_files, valid_files in ret:
            fold_num += 1
            print('Start fold {}'.format(fold_num))

            train_index = train['name'].isin(train_files)
            valid_index = train['name'].isin(valid_files)
            X_train = train.loc[train_index]
            X_valid = train.loc[valid_index]
            y_train = X_train['target']
            y_valid = X_valid['target']
            y_train_cat = to_categorical(y_train, len(CLASSES))
            y_valid_cat = to_categorical(y_valid, len(CLASSES))

            class_weight1 = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
            class_weight1[6] *= random.randint(20, 50)
            print('Class weights: {}'.format(class_weight1))
            print('Train data:', X_train.shape, y_train_cat.shape)
            print('Valid data:', X_valid.shape, y_valid_cat.shape)

            # K.set_image_dim_ordering('th')

            cnn_type = 'ZF_random_keras_blender'
            print('Creating and compiling model [{}]...'.format(cnn_type))
            final_model_path = MODELS_PATH + '{}_fold_{}_{}.h5'.format(cnn_type, fold_num, gpu_use)
            cache_model_path = MODELS_PATH + '{}_temp_fold_{}_{}.h5'.format(cnn_type, fold_num, gpu_use)
            model = ZF_random_keras_blender(len(features))

            if random.randint(0, 1) == 0:
                optim_name = 'SGD'
                learning_rate = random.uniform(0.001, 0.005)
            else:
                optim_name = 'Adam'
                learning_rate = random.uniform(0.001, 0.0001)

            batch_size = random.randint(16, 64)
            epochs = 10000
            patience = random.randint(8, 15)
            print('Batch size: {}'.format(batch_size))
            print('Optim: {} Learning rate: {}'.format(optim_name, learning_rate))
            steps_per_epoch = (X_train.shape[0] // batch_size)
            validation_steps = 2*(X_valid.shape[0] // batch_size)
            print('Steps train: {}, Steps valid: {}'.format(steps_per_epoch, validation_steps))

            if optim_name == 'SGD':
                optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
            else:
                optim = Adam(lr=learning_rate)
            model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

            callbacks = [
                EarlyStopping(monitor='val_acc', patience=patience, verbose=0),
                ModelCheckpoint(cache_model_path, monitor='val_acc', save_best_only=True, verbose=0),
            ]

            history = model.fit_generator(generator=batch_generator_train_blender_random_sample(X_train[features].as_matrix().copy(), y_train_cat.copy(), batch_size),
                                      epochs=epochs,
                                      steps_per_epoch=steps_per_epoch,
                                      validation_data=batch_generator_train_blender_random_sample(X_valid[features].as_matrix().copy(), y_valid_cat.copy(), batch_size),
                                      validation_steps=validation_steps,
                                      verbose=2,
                                      max_queue_size=16,
                                      callbacks=callbacks,
                                      class_weight=class_weight1)

            min_loss = min(history.history['val_loss'])
            max_acc = max(history.history['val_acc'])
            print('Loss for fold {}: {:.6f} Train acc: {:.6f}'.format(fold_num, min_loss, max_acc))
            model.load_weights(cache_model_path)
            model.save(final_model_path)

            if 0:
                now = datetime.datetime.now()
                filename = HISTORY_FOLDER_PATH + 'history_{}_{}_{:.4f}_lr_{}_{}_weather.csv'.format(cnn_type, fold_num,
                                                                                                min_loss, learning_rate,
                                                                                                now.strftime(
                                                                                                    "%Y-%m-%d-%H-%M"))
                pd.DataFrame(history.history).to_csv(filename, index=False)

            pred = model.predict(X_valid[features].as_matrix().copy())
            full_preds[valid_index, :] += pred
            counts[valid_index, :] += 1

            pred_index = np.argmax(pred, axis=1)
            score = accuracy_score(y_valid, pred_index)
            print('Fold {} acc: {:.6f}'.format(fold_num, score))
            model_list.append(model)

    full_preds /= counts
    score = accuracy_score(train['target'].values, np.argmax(full_preds, axis=1))

    s = pd.DataFrame(train['name'].values, columns=['name'])
    for a in CLASSES:
        s[a] = 0.0
    s[CLASSES] = full_preds
    s.to_csv(SUBM_PATH + 'subm_raw_{}_train.csv'.format('keras_blender'), index=False)

    print('Default score: {:.6f}'.format(score))
    print('Time: {} sec'.format(time.time() - start_time))

    return score, full_preds, model_list


def predict_with_keras_model(test, features, models_list):
    dtest = test[features].as_matrix().copy()
    full_preds = []
    for m in models_list:
        preds = m.predict(dtest)
        full_preds.append(preds)
    preds = np.array(full_preds).mean(axis=0)
    return preds


def get_readable_date(dt):
    return datetime.datetime.fromtimestamp(dt).strftime('%Y-%m-%d %H:%M:%S')


def run_keras(iter_num):
    train, test, features = read_tables(rescale=False)
    if 'size' in features:
        features.remove('size')

    gbm_type = 'keras_blender'

    score, valid_pred, model_list = create_keras_blender_model(train, features, iter_num)
    preds = predict_with_keras_model(test, features, model_list)

    subm = pd.DataFrame(test['name'].values, columns=['fname'])
    for a in CLASSES:
        subm[a] = 0.0
    subm[CLASSES] = preds
    subm.to_csv(SUBM_PATH + 'subm_raw_{}_test.csv'.format(gbm_type), index=False)

    submission_file = SUBM_PATH + 'subm_{}_test.csv'.format(gbm_type)
    subm['label_index'] = np.argmax(subm[CLASSES].as_matrix(), axis=1)
    subm['camera'] = np.array(CLASSES)[subm['label_index']]
    subm[['fname', 'camera']].to_csv(submission_file, index=False)
    check_subm_distribution(submission_file)
    # check_subm_diff(SUBM_PATH + '0.991_equal_2_pwr_mean_hun_5_prod-ce..csv', submission_file)


if __name__ == '__main__':
    start_time = time.time()
    # Increase iter_num for better precision
    run_keras(12)
    print("Elapsed time overall: %s seconds" % (time.time() - start_time))

# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

'''
Second level model, which uses all previously generated features, based on Keras classifier
'''

if __name__ == '__main__':
    import os
    gpu_use = 2
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


import math
import datetime
from sklearn.metrics import accuracy_score
from a00_common_functions import *
from a60_second_level_xgboost_all_models import read_tables, rename_columns, check_subm_distribution, check_subm_diff, get_kfold_split_xgboost


def batch_generator_train_blender_random_sample(X, y, batch_size):
    rng = list(range(X.shape[0]))

    while True:
        index1 = random.sample(rng, batch_size)
        input1 = X[index1, :]
        output1 = y[index1]
        yield input1, output1


def ZF_keras_blender_v2(input_features):
    from keras.models import Model
    from keras.layers import Input, Dense
    from keras.layers.core import Dropout

    inputs1 = Input((input_features,))
    x = Dense(input_features, activation='relu')(inputs1)
    x = Dropout(0.5)(x)
    x = Dense(input_features, activation='relu')(x)
    x = Dropout(0.5)(x)
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


def create_keras_blender_model(train, features):
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

    for iter in range(2):
        restore = 0
        num_folds = random.randint(3, 5)
        model_type = random.randint(0, 1)

        print('Train shape:', train.shape)
        ret = get_kfold_split_xgboost(train, num_folds, iter)

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

            print('Train data:', X_train.shape, y_train_cat.shape)
            print('Valid data:', X_valid.shape, y_valid_cat.shape)

            # K.set_image_dim_ordering('th')
            if model_type == 1:
                cnn_type = 'ZF_keras_blender_v2'
                print('Creating and compiling model [{}]...'.format(cnn_type))
                final_model_path = MODELS_PATH + '{}_fold_{}.h5'.format(cnn_type, fold_num)
                cache_model_path = MODELS_PATH + '{}_temp_fold_{}.h5'.format(cnn_type, fold_num)
                model = ZF_keras_blender_v2(len(features))
            else:
                cnn_type = 'ZF_keras_blender_v3'
                print('Creating and compiling model [{}]...'.format(cnn_type))
                final_model_path = MODELS_PATH + '{}_fold_{}.h5'.format(cnn_type, fold_num)
                cache_model_path = MODELS_PATH + '{}_temp_fold_{}.h5'.format(cnn_type, fold_num)
                model = ZF_keras_blender_v3(len(features))

            optim_name = 'Adam'
            batch_size = 48
            learning_rate = 0.00005
            epochs = 10000
            patience = 50
            print('Batch size: {}'.format(batch_size))
            print('Learning rate: {}'.format(learning_rate))
            steps_per_epoch = (X_train.shape[0] // batch_size)
            validation_steps = 2*(X_valid.shape[0] // batch_size)
            print('Steps train: {}, Steps valid: {}'.format(steps_per_epoch, validation_steps))

            if optim_name == 'SGD':
                optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
            else:
                optim = Adam(lr=learning_rate)
            model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

            callbacks = [
                EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
                ModelCheckpoint(cache_model_path, monitor='val_loss', save_best_only=True, verbose=0),
            ]

            history = model.fit_generator(generator=batch_generator_train_blender_random_sample(X_train[features].as_matrix().copy(), y_train_cat.copy(), batch_size),
                                      epochs=epochs,
                                      steps_per_epoch=steps_per_epoch,
                                      validation_data=batch_generator_train_blender_random_sample(X_valid[features].as_matrix().copy(), y_valid_cat.copy(), batch_size),
                                      validation_steps=validation_steps,
                                      verbose=2,
                                      max_queue_size=16,
                                      callbacks=callbacks)

            min_loss = min(history.history['val_loss'])
            print('Minimum loss for given fold: ', min_loss)
            model.load_weights(cache_model_path)
            model.save(final_model_path)

            if 0:
                now = datetime.datetime.now()
                filename = HISTORY_FOLDER_PATH + 'history_{}_{}_{:.4f}_lr_{}_{}_weather.csv'.format(cnn_type, fold_num,
                                                                                                min_loss, learning_rate,
                                                                                                now.strftime(
                                                                                                    "%Y-%m-%d-%H-%M"))
                pd.DataFrame(history.history).to_csv(filename, index=False)

            pred = model.predict(X_valid.as_matrix().copy())
            full_preds[valid_index, :] += pred
            counts[valid_index, :] += 1

            pred_index = np.argmax(pred, axis=1)
            score = accuracy_score(y_valid, pred_index)
            print('Fold {} acc: {}'.format(fold_num, score))
            model_list.append(model)

    full_preds /= counts
    score = accuracy_score(train['target'].values, np.argmax(full_preds, axis=1))

    s = pd.DataFrame(train['name'].values, columns=['name'])
    for a in CLASSES:
        s[a] = 0.0
    s[CLASSES] = full_preds
    s.to_csv(SUBM_PATH + 'ensemble_res/subm_{}_train.csv'.format('keras_blender'), index=False)

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


def run_keras():
    train, test, features = read_tables()
    gbm_type = 'keras_blender'

    if 1:
        score, valid_pred, model_list = create_keras_blender_model(train, features)
        save_in_file((score, valid_pred, model_list), MODELS_PATH + 'keras_last_run_models.pklz'.format())
    else:
        score, valid_pred, model_list = load_from_file(MODELS_PATH + 'keras_last_run_models.pklz'.format())

    preds = predict_with_keras_model(test, features, model_list)

    subm = pd.DataFrame(test['name'].values, columns=['fname'])
    for a in CLASSES:
        subm[a] = 0.0
    subm[CLASSES] = preds
    subm.to_csv(SUBM_PATH + 'ensemble_res/subm_raw_{}_test.csv'.format(gbm_type), index=False)

    submission_file = SUBM_PATH + 'ensemble_res/subm_{}_test.csv'.format(gbm_type)
    subm['label_index'] = np.argmax(subm[CLASSES].as_matrix(), axis=1)
    subm['camera'] = np.array(CLASSES)[subm['label_index']]
    subm[['fname', 'camera']].to_csv(submission_file, index=False)
    check_subm_distribution(submission_file)
    check_subm_diff(SUBM_PATH + '0.991_equal_2_pwr_mean_hun_5_prod-ce..csv', submission_file)


if __name__ == '__main__':
    start_time = time.time()
    run_keras()
    print("Elapsed time overall: %s seconds" % (time.time() - start_time))


'''

'''
# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


import datetime
from sklearn.metrics import accuracy_score
from a00_common_functions import *
from a60_second_level_xgboost_all_models import read_tables, rename_columns, check_subm_distribution, check_subm_diff, get_kfold_split_xgboost
from sklearn.utils import class_weight


def print_importance(features, gbm, prnt=True):
    max_report = 100
    importance_arr = sorted(list(zip(features, gbm.feature_importance())), key=lambda x: x[1], reverse=True)
    s1 = 'Importance TOP {}: '.format(max_report)
    for d in importance_arr[:max_report]:
        s1 += str(d) + ', '
    if prnt:
        print(s1)
    return importance_arr


def create_lightgbm_model(train, features, iter1):
    import lightgbm as lgb
    print('LightGBM version: {}'.format(lgb.__version__))
    start_time = time.time()

    rescaled = len(train)
    model_list = []
    full_preds = np.zeros((rescaled, len(CLASSES)), dtype=np.float32)
    counts = np.zeros((rescaled, len(CLASSES)), dtype=np.float32)

    for iter in range(100):

        # Debug
        num_folds = random.randint(3, 5)
        random_state = 10
        rs = 69
        learning_rate = random.uniform(0.01, 0.05)
        num_leaves = random.randint(31, 63)
        feature_fraction = 0.95
        bagging_fraction = 0.95
        boosting_type = 'gbdt'
        # boosting_type = 'dart'
        min_data_in_leaf = random.randint(128, 256)
        max_bin = 255
        bagging_freq = 0
        drop_rate = 0.05
        skip_drop = 0.5
        max_drop = 1

        params = {
            'task': 'train',
            'boosting_type': boosting_type,
            'objective': 'multiclass',
            'num_class': 10,
            'metric': {'multi_logloss'},
            'device': 'cpu',
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'feature_fraction': feature_fraction,
            'bagging_fraction': bagging_fraction,
            'min_data_in_leaf': min_data_in_leaf,
            'bagging_freq': bagging_freq,
            'max_bin': max_bin,
            'drop_rate': drop_rate,
            'skip_drop': skip_drop,
            'max_drop': max_drop,
            'feature_fraction_seed': random_state + iter,
            'bagging_seed': random_state + iter,
            'data_random_seed': random_state + iter,
            'verbose': 0,
            'num_threads': 9,
        }
        log_str = 'LightGBM iter {}. PARAMS: {}'.format(iter, sorted(params.items()))
        print(log_str)
        num_boost_round = 10000
        early_stopping_rounds = 50

        print('Train shape:', train.shape)
        ret = get_kfold_split_xgboost(train, num_folds, iter1 + iter)

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

            print('Train data:', X_train.shape)
            print('Valid data:', X_valid.shape)

            if 1:
                sample_weight_train = class_weight.compute_sample_weight('balanced', y_train)
                sample_weight_valid = class_weight.compute_sample_weight('balanced', y_valid)
                class_weight1 = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
                coeff1 = random.randint(1, 1)
                sample_weight_train[y_train == CLASSES.index('LG-Nexus-5x')] *= coeff1
                sample_weight_valid[y_valid == CLASSES.index('LG-Nexus-5x')] *= coeff1
                # print(sample_weight1)
                print('Class weights train: {}'.format(np.unique(sample_weight_train)))
                print('Class weights valid: {}'.format(np.unique(sample_weight_valid)))
                # print(class_weight1)
                # exit()

            lgb_train = lgb.Dataset(X_train[features].as_matrix(), y_train, weight=sample_weight_train)
            lgb_eval = lgb.Dataset(X_valid[features].as_matrix(), y_valid, weight=sample_weight_valid, reference=lgb_train)

            gbm = lgb.train(params, lgb_train, num_boost_round=num_boost_round,
                            early_stopping_rounds=early_stopping_rounds, valid_sets=[lgb_eval], verbose_eval=True)

            print_importance(features, gbm, True)
            model_list.append(gbm)

            print("Validating...")
            pred = gbm.predict(X_valid[features].as_matrix(), num_iteration=gbm.best_iteration)
            full_preds[valid_index, :] += pred
            counts[valid_index, :] += 1

            pred_index = np.argmax(pred, axis=1)
            score = accuracy_score(y_valid, pred_index)
            print('Fold {} acc: {}'.format(fold_num, score))

    full_preds /= counts
    score = accuracy_score(train['target'].values, np.argmax(full_preds, axis=1))

    s = pd.DataFrame(train['name'].values, columns=['name'])
    for a in CLASSES:
        s[a] = 0.0
    s[CLASSES] = full_preds
    s.to_csv(SUBM_PATH + 'ensemble_res/subm_raw_{}_{}_train.csv'.format('lightgbm', iter1), index=False)

    print('Default score: {:.6f}'.format(score))
    print('Time: {} sec'.format(time.time() - start_time))

    return score, full_preds, model_list


def predict_with_lightgbm_model(test, features, models_list):
    dtest = test[features].as_matrix()
    full_preds = []
    total = 0
    for m in models_list:
        total += 1
        print('Process test model: {}'.format(total))
        preds = m.predict(dtest, num_iteration=m.best_iteration)
        full_preds.append(preds)
    preds = np.array(full_preds).mean(axis=0)
    return preds


def get_readable_date(dt):
    return datetime.datetime.fromtimestamp(dt).strftime('%Y-%m-%d %H:%M:%S')


def run_lightgbm(lr, iter1):
    train, test, features = read_tables(rescale=False)
    gbm_type = 'lightgbm'

    if 1:
        score, valid_pred, model_list = create_lightgbm_model(train, features, iter1)
        save_in_file((score, valid_pred, model_list), MODELS_PATH + 'lightgbm_last_run_models_{}.pklz'.format(iter1))
    else:
        score, valid_pred, model_list = load_from_file(MODELS_PATH + 'lightgbm_last_run_models_{}.pklz'.format(iter1))

    preds = predict_with_lightgbm_model(test, features, model_list)

    subm = pd.DataFrame(test['name'].values, columns=['fname'])
    for a in CLASSES:
        subm[a] = 0.0
    subm[CLASSES] = preds
    subm.to_csv(SUBM_PATH + 'ensemble_res/subm_raw_{}_{}_test.csv'.format(gbm_type, iter1), index=False)

    submission_file = SUBM_PATH + 'ensemble_res/subm_{}_{}_test.csv'.format(gbm_type, iter1)
    subm['label_index'] = np.argmax(subm[CLASSES].as_matrix(), axis=1)
    subm['camera'] = np.array(CLASSES)[subm['label_index']]
    subm[['fname', 'camera']].to_csv(submission_file, index=False)
    check_subm_distribution(submission_file)
    check_subm_diff(SUBM_PATH + '0.991_equal_2_pwr_mean_hun_5_prod-ce..csv', submission_file)


if __name__ == '__main__':
    start_time = time.time()
    run_lightgbm(0.010, 2)
    print("Elapsed time overall: %s seconds" % (time.time() - start_time))


'''
Single:
HTC-1-M7: [133, 131]
iPhone-6: [134, 132]
Motorola-Droid-Maxx: [132, 134]
Motorola-X: [133, 132]
Samsung-Galaxy-S4: [134, 131]
iPhone-4s: [134, 133]
LG-Nexus-5x: [88, 123]
Motorola-Nexus-6: [153, 140]
Samsung-Galaxy-Note3: [144, 132]
Sony-NEX-7: [135, 132]
Difference in 67 pos from 2640. Percent: 2.54%

20 models:
Default score: 0.991182
HTC-1-M7: [133, 131]
iPhone-6: [133, 132]
Motorola-Droid-Maxx: [132, 134]
Motorola-X: [133, 132]
Samsung-Galaxy-S4: [134, 131]
iPhone-4s: [134, 133]
LG-Nexus-5x: [89, 123]
Motorola-Nexus-6: [152, 140]
Samsung-Galaxy-Note3: [144, 132]
Sony-NEX-7: [136, 132]
Difference in 65 pos from 2640. Percent: 2.46%

1 model + rescale
HTC-1-M7: [133, 131]
iPhone-6: [132, 132]
Motorola-Droid-Maxx: [131, 133]
Motorola-X: [133, 132]
Samsung-Galaxy-S4: [134, 131]
iPhone-4s: [134, 132]
LG-Nexus-5x: [98, 126]
Motorola-Nexus-6: [147, 139]
Samsung-Galaxy-Note3: [142, 132]
Sony-NEX-7: [136, 132]
Difference in 53 pos from 2640. Percent: 2.01%

1 model + rescale
HTC-1-M7: [133, 131]
iPhone-6: [132, 132]
Motorola-Droid-Maxx: [132, 133]
Motorola-X: [134, 132]
Samsung-Galaxy-S4: [132, 131]
iPhone-4s: [135, 132]
LG-Nexus-5x: [109, 129]
Motorola-Nexus-6: [142, 138]
Samsung-Galaxy-Note3: [138, 132]
Sony-NEX-7: [133, 130]
Difference in 49 pos from 2640. Percent: 1.86%

20 models + rescale
HTC-1-M7: [133, 131]
iPhone-6: [132, 132]
Motorola-Droid-Maxx: [132, 133]
Motorola-X: [133, 132]
Samsung-Galaxy-S4: [132, 131]
iPhone-4s: [135, 132]
LG-Nexus-5x: [125, 134]
Motorola-Nexus-6: [128, 134]
Samsung-Galaxy-Note3: [137, 132]
Sony-NEX-7: [133, 129]
Difference in 54 pos from 2640. Percent: 2.05%

100 models
HTC-1-M7: [133, 131]
iPhone-6: [132, 132]
Motorola-Droid-Maxx: [132, 133]
Motorola-X: [133, 132]
Samsung-Galaxy-S4: [132, 131]
iPhone-4s: [133, 132]
LG-Nexus-5x: [125, 134]
Motorola-Nexus-6: [129, 134]
Samsung-Galaxy-Note3: [137, 132]
Sony-NEX-7: [134, 129]
Difference in 53 pos from 2640. Percent: 2.01%
'''

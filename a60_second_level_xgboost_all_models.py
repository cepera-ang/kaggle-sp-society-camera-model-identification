# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


import datetime
from operator import itemgetter
from a00_common_functions import *
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight


def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance


def get_kfold_split_xgboost(train, num_folds=4, seed=66):
    uni_names = pd.unique(train['name'])
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    ret = []
    for train_index, test_index in kf.split(range(len(uni_names))):
        train_files = uni_names[train_index]
        test_files = uni_names[test_index]
        ret.append((train_files, test_files))
    return ret


def create_xgboost_model(train, features, eta_value, depth, iter1):
    import xgboost as xgb
    print('XGBoost version: {}'.format(xgb.__version__))
    start_time = time.time()

    rescaled = len(train)
    model_list = []
    full_preds = np.zeros((rescaled, len(CLASSES)), dtype=np.float32)
    counts = np.zeros((rescaled, len(CLASSES)), dtype=np.float32)

    for zz in range(20):
        print('Iteration: {}'.format(zz))
        num_folds = random.randint(3, 5)
        eta = random.uniform(0.1, 0.3)
        max_depth = random.randint(1, 2)
        subsample = 0.9
        colsample_bytree = 0.9
        if random.randint(0, 1) == 0:
            eval_metric = 'mlogloss'
        else:
            eval_metric = 'merror'
        unique_target = np.array(sorted(train['target'].unique()))
        print('Target length: {}: {}'.format(len(unique_target), unique_target))

        log_str = 'XGBoost iter {}. FOLDS: {} METRIC: {} ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(0,
                                                                                                               num_folds,
                                                                                                               eval_metric,
                                                                                                               eta,
                                                                                                               max_depth,
                                                                                                               subsample,
                                                                                                               colsample_bytree)
        print(log_str)
        params = {
            "objective": "multi:softprob",
            "num_class": len(unique_target),
            "booster": "gbtree",
            "eval_metric": eval_metric,
            "eta": eta,
            "tree_method": 'exact',
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "silent": 1,
            "seed": 2017,
            "nthread": 6,
            # 'gpu_id': 0,
            # 'updater': 'grow_gpu_hist',
        }
        num_boost_round = 1500
        early_stopping_rounds = 25

        print('Train shape:', train.shape)
        ret = get_kfold_split_xgboost(train, num_folds, iter1+zz)

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
                coeff1 = random.randint(100, 200)
                coeff2 = random.randint(2, 4)
                sample_weight_train[y_train == CLASSES.index('LG-Nexus-5x')] *= coeff1
                sample_weight_valid[y_valid == CLASSES.index('LG-Nexus-5x')] *= coeff1
                sample_weight_train[y_train == CLASSES.index('Motorola-Nexus-6')] *= coeff2
                sample_weight_valid[y_valid == CLASSES.index('Motorola-Nexus-6')] *= coeff2
                # print(sample_weight1)
                print('Class weights train: {}'.format(np.unique(sample_weight_train)))
                print('Class weights valid: {}'.format(np.unique(sample_weight_valid)))


            dtrain = xgb.DMatrix(X_train[features].as_matrix(), y_train, weight=sample_weight_train)
            dvalid = xgb.DMatrix(X_valid[features].as_matrix(), y_valid, weight=sample_weight_valid)

            watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
            gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist,
                            early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
            model_list.append(gbm)

            imp = get_importance(gbm, features)
            print('Importance: {}'.format(imp))

            print("Validating...")
            pred = gbm.predict(dvalid, ntree_limit=gbm.best_iteration + 1)
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
    s.to_csv(SUBM_PATH + 'ensemble_res/subm_raw_{}_{}_train.csv'.format('xgboost', iter1), index=False)

    if 0:
        s['target'] = train['target']
        norm_score = 0
        for i in range(len(CLASSES)):
            part = s[s['target'] == i]
            pscore = accuracy_score(part['target'].values, np.argmax(part[CLASSES].as_matrix(), axis=1))
            print('{} acc {}'.format(CLASSES[i], pscore))
            norm_score += pscore

    print('Default score: {:.6f}'.format(score))
    # print('Normalized score: {:.6f}'.format(norm_score/len(CLASSES)))
    print('Time: {} sec'.format(time.time() - start_time))

    return score, full_preds, model_list


def predict_with_xgboost_model(test, features, models_list):
    import xgboost as xgb

    dtest = xgb.DMatrix(test[features].as_matrix())
    full_preds = []
    for m in models_list:
        preds = m.predict(dtest, ntree_limit=m.best_iteration + 1)
        full_preds.append(preds)
    preds = np.array(full_preds).mean(axis=0)
    return preds


def get_readable_date(dt):
    return datetime.datetime.fromtimestamp(dt).strftime('%Y-%m-%d %H:%M:%S')


def rescale_train(train):
    uni = (train['target'].unique())
    max_label = 0
    for u in uni:
        l = len(train[train['target'] == u])
        if l > max_label:
            max_label = l
        print(CLASSES[u], l)
    print('Max label: {}'.format(max_label))

    new_train = []
    for u in uni:
        part = train[train['target'] == u]
        l = len(part)
        incr = max_label // l
        print(CLASSES[u], incr)
        for i in range(incr):
            new_train.append(part.copy())
        if max_label > incr*l:
            print('A: {}'.format(max_label-incr*l))
            new_train.append(part[:max_label-incr*l].copy())
        if CLASSES[u] == 'LG-Nexus-5x':
            for ii in range(200):
                new_train.append(part.copy())

    train = pd.concat(new_train, axis=0)
    train.reset_index(drop=True, inplace=True)
    print(len(train))
    return train


def rename_columns(tbl, suffix):
    for c in CLASSES:
        tbl.rename(columns={c: c + suffix}, inplace=True)
    return tbl


def read_tables(rescale=True):
    train_list = [
        (SUBM_PATH + 'ensemble_big/976_tta_8_densenet201_antorsaegen_62_0.98271093_train.hdf5', 1),
        (SUBM_PATH + 'ensemble_big/977_tta_8_resnet50_antorsaegen_119_val_0.9815_train.hdf5', 1),
        (SUBM_PATH + 'ensemble_big/DenseNet201_do0.3_doc0.0_avg-epoch072-val_acc0.981250_train.hdf5', 1),
        (SUBM_PATH + 'ensemble_big/InceptionResNetV2_do0.1_avg-epoch154-val_acc0.965625_train.hdf5', 1),
        (SUBM_PATH + 'ensemble_big/Xception_do0.3_avg-epoch079-val_acc0.991667_train.hdf5', 1),
        (SUBM_PATH + 'ensemble_big/984_tta_8_densenet201_29_0.98624_train.hdf5', 1),
    ]

    full_train = []
    full_test = []
    total = 0
    for train_val in train_list:
        print('Read: {}'.format(train_val))
        p = train_val[0]
        train = pd.read_hdf(p, "prob")
        train = train.groupby(["name"]).aggregate('mean')
        train['name'] = train.index
        main_manip = []
        for nm in list(train['name'].values):
            if 'manip' in nm:
                main_manip.append(1)
            else:
                main_manip.append(0)
        train['manip'] = main_manip
        train = rename_columns(train, '_{}'.format(total))
        train.drop('aug', axis=1, inplace=True)
        full_train.append(train)

        test = pd.read_hdf(p[:-11] + '_test.hdf5')
        test = test.groupby(["name"]).aggregate('mean')
        test['name'] = test.index
        main_manip = []
        for nm in list(test['name'].values):
            if 'manip' in nm:
                main_manip.append(1)
            else:
                main_manip.append(0)
        test['manip'] = main_manip
        test = rename_columns(test, '_{}'.format(total))
        full_test.append(test)

        total += 1

    train = full_train[0]
    test = full_test[0]
    for i in range(1, len(full_train)):
        train = pd.merge(train, full_train[i], on=['name', 'manip'], left_index=True)
    for i in range(1, len(full_test)):
        test = pd.merge(test, full_test[i], on=['name', 'manip'], left_index=True)
    print('Train length: {}'.format(len(train)))
    print('Test length: {}'.format(len(test)))
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    names_list = list(train['name'].values)
    target = []
    for nm in names_list:
        clss = nm.split('_')[0]
        try:
            target.append(CLASSES.index(clss))
        except:
            print(clss, nm)
            exit()
    train['target'] = target

    if 0:
        sz_train = pd.read_csv(SUBM_PATH + 'ensemble_big/sizes/test_new_size_byte.csv')
        train = pd.merge(train, sz_train, on='name', left_index=True)
        sz_test = pd.read_csv(SUBM_PATH + 'ensemble_big/sizes/test_prod_size_byte.csv')
        test = pd.merge(test, sz_test, on='name', left_index=True)

    features = list(train.columns.values)
    features.remove('name')
    features.remove('target')
    print('Features [{}]: {}'.format(len(features), features))

    is_null = train.isnull().values.any()
    if is_null:
        print('Train contains null!')
        exit()
    is_null = test.isnull().values.any()
    if is_null:
        print('Test contains null!')
        exit()

    uni = pd.value_counts(train['target'])
    print('Target counts:')
    print(uni)
    if rescale is True:
        print('Rescale it!')
        train = rescale_train(train)
        uni = pd.value_counts(train['target'])
        print('Target counts:')
        print(uni)

    return train, test, features


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


def run_xgboost(eta, depth, iter1):
    train, test, features = read_tables(rescale=False)
    gbm_type = 'xgboost'

    if 1:
        score, valid_pred, model_list = create_xgboost_model(train, features, eta, depth, iter1)
        save_in_file((score, valid_pred, model_list), MODELS_PATH + 'xgboost_last_run_models_{}.pklz'.format(iter1))
    else:
        score, valid_pred, model_list = load_from_file(MODELS_PATH + 'xgboost_last_run_models_{}.pklz'.format(iter1))

    preds = predict_with_xgboost_model(test, features, model_list)
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
    # preproc_manip_densnet_201()
    run_xgboost(0.2, 1, 2)
    print("Elapsed time overall: %s seconds" % (time.time() - start_time))


'''
Default score: 0.991120
HTC-1-M7: [133, 131]
iPhone-6: [134, 132]
Motorola-Droid-Maxx: [134, 133]
Motorola-X: [132, 132]
Samsung-Galaxy-S4: [135, 131]
iPhone-4s: [135, 133]
LG-Nexus-5x: [89, 122]
Motorola-Nexus-6: [151, 140]
Samsung-Galaxy-Note3: [142, 133]
Sony-NEX-7: [135, 133]
Difference in 68 pos from 2640. Percent: 2.58%

Default run
Default score: 0.990747
HTC-1-M7: [134, 131]
iPhone-6: [134, 132]
Motorola-Droid-Maxx: [132, 133]
Motorola-X: [132, 132]
Samsung-Galaxy-S4: [135, 131]
iPhone-4s: [134, 133]
LG-Nexus-5x: [91, 124]
Motorola-Nexus-6: [151, 139]
Samsung-Galaxy-Note3: [142, 132]
Sony-NEX-7: [135, 133]
Difference in 63 pos from 2640. Percent: 2.39% LB: 0.982

Rescale (uniform):
Default score: 0.987413
Time: 465.3627233505249 sec
HTC-1-M7: [133, 131]
iPhone-6: [134, 132]
Motorola-Droid-Maxx: [131, 133]
Motorola-X: [133, 132]
Samsung-Galaxy-S4: [134, 131]
iPhone-4s: [134, 133]
LG-Nexus-5x: [94, 124]
Motorola-Nexus-6: [148, 138]
Samsung-Galaxy-Note3: [142, 132]
Sony-NEX-7: [137, 134]
Difference in 58 pos from 2640. Percent: 2.20%

Rescale (uniform + Nexus*4):
Default score: 0.984956
Time: 465.3627233505249 sec
HTC-1-M7: [133, 131]
iPhone-6: [134, 132]
Motorola-Droid-Maxx: [132, 133]
Motorola-X: [133, 132]
Samsung-Galaxy-S4: [133, 131]
iPhone-4s: [134, 132]
LG-Nexus-5x: [99, 126]
Motorola-Nexus-6: [144, 138]
Samsung-Galaxy-Note3: [141, 132]
Sony-NEX-7: [137, 133]
Difference in 52 pos from 2640. Percent: 1.97%

Rescale (uniform + Nexus*50):
Default score: 0.986832
Time: 450.1165554523468 sec
HTC-1-M7: [133, 131]
iPhone-6: [133, 132]
Motorola-Droid-Maxx: [133, 133]
Motorola-X: [133, 132]
Samsung-Galaxy-S4: [133, 131]
iPhone-4s: [135, 132]
LG-Nexus-5x: [116, 128]
Motorola-Nexus-6: [130, 137]
Samsung-Galaxy-Note3: [139, 132]
Sony-NEX-7: [135, 132]
Difference in 44 pos from 2640. Percent: 1.67%

Rescale (uniform + Nexus*200):
Default score: 0.992147
Time: 990.5899105072021 sec
HTC-1-M7: [133, 131]
iPhone-6: [133, 132]
Motorola-Droid-Maxx: [132, 133]
Motorola-X: [132, 132]
Samsung-Galaxy-S4: [133, 131]
iPhone-4s: [134, 132]
LG-Nexus-5x: [126, 131]
Motorola-Nexus-6: [122, 135]
Samsung-Galaxy-Note3: [138, 132]
Sony-NEX-7: [137, 131]
Difference in 46 pos from 2640. Percent: 1.74%

v1: 20 runs (uniform + Nexus*200)
XGBoost iter 0. FOLDS: 5 METRIC: merror ETA: 0.22576181674273743, MAX_DEPTH: 1, SUBSAMPLE: 0.9, COLSAMPLE_BY_TREE: 0.9
HTC-1-M7: [132, 131]
iPhone-6: [133, 132]
Motorola-Droid-Maxx: [133, 133]
Motorola-X: [132, 132]
Samsung-Galaxy-S4: [133, 131]
iPhone-4s: [134, 132]
LG-Nexus-5x: [132, 136]
Motorola-Nexus-6: [117, 130]
Samsung-Galaxy-Note3: [138, 132]
Sony-NEX-7: [136, 131]
Difference in 50 pos from 2640. Percent: 1.89%
'''
# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a00_common_functions import *
from a60_second_level_xgboost_all_models import check_subm_distribution


def ensemble():
    tables_list = [
        (SUBM_PATH + 'subm_raw_xgboost_test.csv', 1),
        (SUBM_PATH + 'subm_raw_lightgbm_test.csv', 1),
        (SUBM_PATH + 'subm_raw_keras_blender_test.csv', 1),
    ]
    res = None
    res_train = None
    sum_weight = 0
    for t in tables_list:
        w = t[1]
        path = t[0]
        path_train = path[:-9] + '_train.csv'
        s = pd.read_csv(path, index_col='fname')
        s_train = pd.read_csv(path_train, index_col='name')
        if res is None:
            res = w*s
            res_train = w*s_train
        else:
            res += w*s
            res_train += w*s_train
        sum_weight += w
    res[CLASSES] /= sum_weight
    res_train[CLASSES] /= sum_weight

    submission_file = SUBM_PATH + 'final_ensmble_{}_raw_train.csv'.format(len(tables_list))
    res_train.to_csv(submission_file)
    submission_file = SUBM_PATH + 'final_ensmble_{}_raw_test.csv'.format(len(tables_list))
    res.to_csv(submission_file)

    submission_file = SUBM_PATH + 'final_ensmble_{}.csv'.format(len(tables_list))
    subm = res.copy()
    subm['label_index'] = np.argmax(subm[CLASSES].as_matrix(), axis=1)
    subm['camera'] = np.array(CLASSES)[subm['label_index']]
    subm[['camera']].to_csv(submission_file)
    check_subm_distribution(submission_file)
    # check_subm_diff(SUBM_PATH + '0.991_equal_2_pwr_mean_hun_5_prod-ce..csv', submission_file)
    print('Submission file stored in {}'.format(submission_file))


if __name__ == '__main__':
    ensemble()

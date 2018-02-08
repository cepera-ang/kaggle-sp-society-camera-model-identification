# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a00_common_functions import *
from a60_second_level_xgboost_all_models import check_subm_distribution, check_subm_diff


def ensemble():
    tables_list = [
        (SUBM_PATH + 'ensemble_res/run2/subm_raw_xgboost_2_test.csv', 4),
        (SUBM_PATH + 'ensemble_res/run2/subm_raw_lightgbm_2_test.csv', 4),
        (SUBM_PATH + 'ensemble_res/run2/subm_raw_keras_blender_1518039467.941106_test.csv', 1),
        (SUBM_PATH + 'ensemble_res/run2/subm_raw_keras_blender_1518039432.0954723_test.csv', 1),
        (SUBM_PATH + 'ensemble_res/run2/subm_raw_keras_blender_1518039426.1696236_test.csv', 1),
        (SUBM_PATH + 'ensemble_res/run2/subm_raw_keras_blender_1518039421.5379343_test.csv', 1),
    ]
    tables_list = [
        (SUBM_PATH + 'ensemble_res/run3/subm_raw_xgboost_2_test.csv', 1),
        (SUBM_PATH + 'ensemble_res/run3/subm_raw_lightgbm_2_test.csv', 1),
        (SUBM_PATH + 'ensemble_res/run3/subm_raw_keras_blender_1518044605.4652689_test.csv', 1),
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
    check_subm_diff(SUBM_PATH + '0.991_equal_2_pwr_mean_hun_5_prod-ce..csv', submission_file)


if __name__ == '__main__':
    ensemble()

'''
HTC-1-M7: [133, 132]
iPhone-6: [133, 132]
Motorola-Droid-Maxx: [131, 132]
Motorola-X: [132, 132]
Samsung-Galaxy-S4: [132, 131]
iPhone-4s: [133, 132]
LG-Nexus-5x: [128, 134]
Motorola-Nexus-6: [126, 133]
Samsung-Galaxy-Note3: [136, 132]
Sony-NEX-7: [136, 130]
Difference in 44 pos from 2640. Percent: 1.67%
LB: 0.986

Big
HTC-1-M7: [133, 131]
iPhone-6: [133, 132]
Motorola-Droid-Maxx: [132, 133]
Motorola-X: [133, 132]
Samsung-Galaxy-S4: [133, 131]
iPhone-4s: [134, 132]
LG-Nexus-5x: [104, 130]
Motorola-Nexus-6: [144, 136]
Samsung-Galaxy-Note3: [139, 132]
Sony-NEX-7: [135, 131]
Difference in 49 pos from 2640. Percent: 1.86%

'''
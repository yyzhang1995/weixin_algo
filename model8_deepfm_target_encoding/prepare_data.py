# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import *
import os
import gc


# 存储数据的根目录
ROOT_PATH = "./"
# 比赛数据集路径
DATASET_PATH = "../dataset/"
DESTINATION_PATH = "./dataset/"
# 训练集
USER_ACTION = DATASET_PATH + "user_action.csv"
FEED_INFO = DATASET_PATH + "feed_info.csv"
FEED_EMBEDDINGS = DATASET_PATH + "feed_embeddings.csv"
FEED_EMBEDDINGS_PCA = DATASET_PATH + 'feed_embeddings_pca.csv'
# 测试集
TEST_FILE = DATASET_PATH + "test_a.csv"
# 初赛待预测行为列表
ACTION_LIST = ["read_comment", "like", "click_avatar", "forward"]
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]
# FEA_FEED_LIST = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id', 'manual_tag_list',
#                  'machine_tag_list', 'machine_keyword_list']
# 负样本下采样比例(负样本:正样本)
# ACTION_SAMPLE_RATE = {"read_comment": 5, "like": 5, "click_avatar": 5, "forward": 10, "comment": 10, "follow": 10,
#                       "favorite": 10}
# # feed_embedding降维之后的特征数量
# feed_embed_feat_num = 8
# FEA_FEED_LIST += [f"embed{i}" for i in range(feed_embed_feat_num)]

__all__ = ['prepare_data']


def prepare_data(params):
    """

    :param params: dict, 包括特征
    :return:
    """
    # reading params
    embed_columns = params['embed_columns']
    FEA_FEED_LIST = params['FEA_FEED_LIST']
    ACTION_SAMPLE_RATE = params['ACTION_SAMPLE_RATE']

    # loading data
    datasets = {}
    feed_info_df = pd.read_csv(FEED_INFO)
    user_action_df = pd.read_csv(USER_ACTION)[["userid", "date_", "feedid", 'device', 'play', 'stay'] + ACTION_LIST]
    test = pd.read_csv(TEST_FILE)
    test['date_'] = 15

    # preprocessing feed_info
    if 'machine_tag_list' in FEA_FEED_LIST:
        process_machine_tag_list(feed_info_df)
    if 'manual_tag_list' in FEA_FEED_LIST:
        process_manual_tag_list(feed_info_df)
    if 'manual_keyword_list' in FEA_FEED_LIST:
        process_manual_keyword(feed_info_df)
    if 'machine_keyword_list' in FEA_FEED_LIST:
        process_machine_keyword(feed_info_df)
    # add feed feature
    train_test_data = pd.concat([user_action_df, test], axis=0, ignore_index=True)
    train_test_data = train_test_data.merge(feed_info_df[FEA_FEED_LIST], on='feedid', how='left')
    train_test_data['videoplayseconds'] *= 1000
    # 是否观看完视频（其实不用严格按大于关系，也可以按比例，比如观看比例超过0.9就算看完）
    train_test_data['is_finish'] = (train_test_data['play'] >= train_test_data['videoplayseconds']).astype('int8')
    train_test_data['play_times'] = train_test_data['play'] / train_test_data['videoplayseconds']

    print(train_test_data.columns)

    # 减少内存消耗
    train_test_data = reduce_mem(train_test_data, [f for f in train_test_data.columns if f not in ['date_'] +
                                                   ['is_finish', 'play_times', 'play', 'stay'] + ACTION_LIST])

    # 统计在过去5天内的行为
    n_day = 5
    for stat_cols in tqdm([
        ['userid'],
        ['feedid'],
        ['authorid'],
        ['userid', 'authorid']
    ]):
        f = '_'.join(stat_cols)
        stat_df = pd.DataFrame()
        for target_day in range(2, 16):
            left, right = max(target_day - n_day, 1), target_day - 1
            tmp = train_test_data[((train_test_data['date_'] >= left) &
                                   (train_test_data['date_'] <= right))].reset_index(drop=True)
            tmp['date_'] = target_day
            tmp['{}_{}day_count'.format(f, n_day)] = tmp.groupby(stat_cols)['date_'].transform('count').astype(np.int32)
            g = tmp.groupby(stat_cols)
            tmp['{}_{}day_finish_rate'.format(f, n_day)] = g['is_finish'].transform('mean').astype(np.float32)
            feats = ['{}_{}day_count'.format(f, n_day), '{}_{}day_finish_rate'.format(f, n_day)]
            for x in ['play_times', 'play', 'stay']:
                for stat in ['max', 'mean']:
                    tmp['{}_{}day_{}_{}'.format(f, n_day, x, stat)] = g[x].transform(stat)
                    feats.append('{}_{}day_{}_{}'.format(f, n_day, x, stat))
            for action in ACTION_LIST:
                tmp['{}_{}day_{}_sum'.format(f, n_day, action)] = g[action].transform('sum').astype(np.int16)
                tmp['{}_{}day_{}_mean'.format(f, n_day, action)] = g[action].transform('mean').astype(np.float32)
                feats.extend(['{}_{}day_{}_sum'.format(f, n_day, action), '{}_{}day_{}_mean'.format(f, n_day, action)])
            tmp = tmp[stat_cols + feats + ['date_']].drop_duplicates(stat_cols + ['date_']).reset_index(drop=True)
            stat_df = pd.concat([stat_df, tmp], axis=0, ignore_index=True)
            del g, tmp
            gc.collect()
        train_test_data = train_test_data.merge(stat_df, on=stat_cols + ['date_'], how='left')
        del stat_df
        gc.collect()

    train = train_test_data[~train_test_data['read_comment'].isna()].reset_index(drop=True)
    test = train_test_data[train_test_data['read_comment'].isna()].reset_index(drop=True)

    # preprocessing feed_embeddings
    if not os.path.exists(FEED_EMBEDDINGS_PCA):
        feed_embed = pd.read_csv(FEED_EMBEDDINGS)
        feed_embed_pca =process_embed(feed_embed)
        feed_embed_pca.to_csv(FEED_EMBEDDINGS_PCA, index=False)
    else:
        feed_embed_pca = pd.read_csv(FEED_EMBEDDINGS_PCA)
    feed_embed_pca = feed_embed_pca[embed_columns]
    train = train.merge(feed_embed_pca, on='feedid', how='left')
    test = test.merge(feed_embed_pca, on='feedid', how='left')

    test.to_csv(DESTINATION_PATH + f'/test_data.csv', index=False)

    for action in tqdm(ACTION_LIST):
        print(f"prepare data for {action}")
        df_neg = train[train[action] == 0]
        df_neg = df_neg.sample(frac=1.0 / ACTION_SAMPLE_RATE[action], random_state=42, replace=False)
        df_all = pd.concat([df_neg, train[train[action] == 1]])
        # datasets[action] = df_all
        df_all.to_csv(DESTINATION_PATH + f'/train_data_for_{action}.csv', index=False)
        del df_neg, df_all
        gc.collect()


if __name__ == "__main__":
    pass

    # feed_info_df = pd.read_csv(FEED_INFO)[['manual_tag_list']]
    # process_manual_tag_list(feed_info_df)
    # print(feed_info_df.head())

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import *
import os


# 存储数据的根目录
ROOT_PATH = "./"
# 比赛数据集路径
DATASET_PATH = '../dataset/'
# 训练集
USER_ACTION = DATASET_PATH + "user_action.csv"
FEED_INFO = DATASET_PATH + "feed_info.csv"
FEED_EMBEDDINGS = DATASET_PATH + "feed_embeddings.csv"
FEED_EMBEDDINGS_PCA = DATASET_PATH + 'feed_embeddings_pca.csv'
# 生成数据保存位置
DESTINATION = './datasets/'
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
    random_seed = params['random_seed']

    # loading data
    datasets = {}
    feed_info_df = pd.read_csv(FEED_INFO)
    user_action_df = pd.read_csv(USER_ACTION)[["userid", "date_", "feedid", 'device'] + FEA_COLUMN_LIST]
    test = pd.read_csv(TEST_FILE)

    # preprocessing feed_embeddings
    if not os.path.exists(FEED_EMBEDDINGS_PCA):
        feed_embed = pd.read_csv(FEED_EMBEDDINGS)
        feed_embed_pca =process_embed(feed_embed)
        feed_embed_pca.to_csv(FEED_EMBEDDINGS_PCA, index=False)
    else:
        feed_embed_pca = pd.read_csv(FEED_EMBEDDINGS_PCA)
    # embed_columns = ['feedid'] + [f"embed{i}" for i in range(feed_embed_feat_num)]
    feed_embed_pca = feed_embed_pca[embed_columns]
    feed_info_df = pd.merge(feed_info_df, feed_embed_pca, on='feedid', how='left')
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
    train = pd.merge(user_action_df, feed_info_df[FEA_FEED_LIST], on='feedid', how='left')
    test = pd.merge(test, feed_info_df[FEA_FEED_LIST], on='feedid', how='left')
    test["videoplayseconds"] = np.log(test["videoplayseconds"] + 1.0)
    # datasets['test'] = test
    test.to_csv(DESTINATION + f'/test_data_seed{random_seed}.csv', index=False)
    for action in tqdm(ACTION_LIST):
        print(f"prepare data for {action}")
        tmp = train.drop_duplicates(['userid', 'feedid', action], keep='last')
        df_neg = tmp[tmp[action] == 0]
        df_neg = df_neg.sample(frac=1.0 / ACTION_SAMPLE_RATE[action], random_state=random_seed, replace=False)
        df_all = pd.concat([df_neg, tmp[tmp[action] == 1]])
        df_all["videoplayseconds"] = np.log(df_all["videoplayseconds"] + 1.0)
        # datasets[action] = df_all
        df_all.to_csv(DESTINATION + f'/train_data_for_{action}_seed{random_seed}.csv', index=False)
    # return datasets


if __name__ == "__main__":
    prepare_data()


    # feed_info_df = pd.read_csv(FEED_INFO)[['manual_tag_list']]
    # process_manual_tag_list(feed_info_df)
    # print(feed_info_df.head())

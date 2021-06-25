from deepFM import *
from prepare_data import *
import pandas as pd
import os


DESTINATION = './datasets/'


def generate_data(seed=42):
    # 设置样本特征
    params = {}
    FEA_FEED_LIST = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id', 'manual_tag_list',
                     'machine_tag_list', 'machine_keyword_list', 'manual_keyword_list']
    # 负样本下采样比例(负样本:正样本)
    ACTION_SAMPLE_RATE = {"read_comment": 5, "like": 5, "click_avatar": 5, "forward": 10, "comment": 10, "follow": 10,
                          "favorite": 10}
    # feed_embedding降维之后的特征数量
    feed_embed_feat_num = 8
    FEA_FEED_LIST += [f"embed{i}" for i in range(feed_embed_feat_num)]
    embed_columns = ['feedid'] + [f"embed{i}" for i in range(feed_embed_feat_num)]

    params['FEA_FEED_LIST'] = FEA_FEED_LIST
    params['ACTION_SAMPLE_RATE'] = ACTION_SAMPLE_RATE
    params['embed_columns'] = embed_columns
    params['random_seed'] = seed
    # 获取数据
    prepare_data(params)


def train_data(seed=42):
    # 设置训练超参数
    FEA_FEED_LISTs = {'read_comment': ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id',
                                       'machine_tag_list', 'manual_tag_list', 'machine_keyword_list'],
                      'like': ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id',
                               'machine_tag_list', 'manual_tag_list'],
                      'click_avatar': ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id',
                                       'machine_tag_list', 'manual_tag_list', 'machine_keyword_list',
                                       'manual_keyword_list'],
                      'forward': ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id',
                                  'machine_tag_list', 'manual_tag_list', 'machine_keyword_list']}
    embedding_dims = {'read_comment': 4, 'like': 4, 'click_avatar': 4, 'forward': 4}
    dnn_hidden_units = {'read_comment': (256, 128), 'like': (512, 256),
                        'click_avatar': (256, 128), 'forward': (256, 128)}
    epochs = {'read_comment': 3, 'like': 2, 'click_avatar': 2, 'forward': 2}
    feed_embed_feat_nums = {'read_comment': 4, 'like': 8, 'click_avatar': 8, 'forward': 8}
    k = 5
    train_params = {}
    train_params['epochs'] = epochs
    train_params['FEA_FEED_LISTs'] = FEA_FEED_LISTs
    train_params['embedding_dims'] = embedding_dims
    train_params['dnn_hidden_units'] = dnn_hidden_units
    train_params['feed_embed_feat_nums'] = feed_embed_feat_nums
    train_params['k'] = k

    train_params['random_seed'] = seed

    res = train_deepfm(train_params)
    return res


def bagging(L, seeds):
    assert L == len(seeds)
    # 设置任务类型
    ACTION_LIST = ["read_comment", "like", "click_avatar", "forward"]
    if seeds is None:
        seeds = list(range(L))

    for i in range(L):
        print("generating data for seed %d" % seeds[i])
        if os.path.exists(DESTINATION + f'/test_data_seed{seeds[i]}.csv'):
            print("datasets for %d already exists!" % (seeds[i]))
            continue
        generate_data(seeds[i])

    for i in range(L):
        print("training predictor %d" % (i + 1))
        train_data(seeds[i])

    res = None
    for i in range(L):
        temp = pd.read_csv("./results/" + f"submit_advance_deepfm_seed{seeds[i]}.csv")
        if res is None:
            res = temp
        else:
            for action in ACTION_LIST:
                res[action] += temp[action]
    for action in ACTION_LIST:
        res[action] = res[action] / L
    res.to_csv("./results/submit_bagging_deepfm.csv", index=False)


if __name__ == '__main__':
    bagging(8, [42, 1, 2, 3, 4, 5, 6, 7])
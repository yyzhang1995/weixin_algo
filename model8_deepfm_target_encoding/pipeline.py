from deepFM import *
from prepare_data import *
import os


def prepare_datasets():
    # 设置样本特征
    params = {}
    FEA_FEED_LIST = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id', 'manual_tag_list',
                     'machine_tag_list', 'machine_keyword_list', 'manual_keyword_list']
    # 负样本下采样比例(负样本:正样本)
    ACTION_SAMPLE_RATE = {"read_comment": 5, "like": 5, "click_avatar": 5, "forward": 10, "comment": 10, "follow": 10,
                          "favorite": 10}
    # feed_embedding降维之后的特征数量
    feed_embed_feat_num = 8
    # FEA_FEED_LIST += [f"embed{i}" for i in range(feed_embed_feat_num)]
    embed_columns = ['feedid'] + [f"embed{i}" for i in range(feed_embed_feat_num)]

    params['FEA_FEED_LIST'] = FEA_FEED_LIST
    params['ACTION_SAMPLE_RATE'] = ACTION_SAMPLE_RATE
    params['embed_columns'] = embed_columns
    # 获取数据
    prepare_data(params)


def train_datasets():
    mode = 'train'
    ACTION_LIST = ['read_comment', 'like', 'click_avatar', 'forward']
    # 设置训练超参数
    FEA_FEED_LISTs = {'read_comment': ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id',
                                       'machine_tag_list', 'manual_tag_list', 'machine_keyword_list'],
                      'like': ['feedid', 'authorid', 'videoplayseconds',
                               'machine_tag_list', 'manual_tag_list'],
                      'click_avatar': ['feedid', 'authorid', 'videoplayseconds',
                                       'machine_tag_list', 'manual_tag_list', 'machine_keyword_list',
                                       'manual_keyword_list'],
                      'forward': ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id',
                                  'machine_tag_list', 'manual_tag_list', 'machine_keyword_list']}
    day_num = 5
    FEA_ACTIONS = {}
    ALL_FEA_ACTION = []
    for main_features in [['userid'], ['feedid'], ['authorid'], ['userid', 'authorid']]:
        f = '_'.join(main_features)
        ALL_FEA_ACTION += [f"{f}_{day_num}day_{ope}" for ope in ['count', 'finish_rate',
                                                                 'play_times_max', 'play_times_mean',
                                                                 'play_max', 'play_mean', 'stay_max', 'stay_mean']] + \
                          [f"{f}_{day_num}day_{action}_{ope}" for action in ACTION_LIST for ope in ['sum', 'mean']]
    for action in ACTION_LIST:
        FEA_ACTION = []
        for main_features in [['userid'], ['feedid'], ['authorid'], ['userid', 'authorid']]:
            f = '_'.join(main_features)
            FEA_ACTION += [f"{f}_{day_num}day_{ope}" for ope in ['count', 'finish_rate',
                                                                 'play_times_max', 'play_times_mean',
                                                                 'play_max', 'play_mean', 'stay_max', 'stay_mean']] +\
                          [f"{f}_{day_num}day_{action}_{ope}" for ope in ['sum', 'mean']]
        FEA_ACTIONS[action] = FEA_ACTION
    FEA_ACTIONS['read_comment'] = FEA_ACTIONS['like'] = FEA_ACTIONS['forward'] = ALL_FEA_ACTION
    embedding_dims = {'read_comment': 4, 'like': 4, 'click_avatar': 4, 'forward': 4}
    dnn_hidden_units = {'read_comment': (256, 128), 'like': (512, 256),
                        'click_avatar': (256, 128), 'forward': (256, 128)}
    epochs = {'read_comment': 3, 'like': 2, 'click_avatar': 2, 'forward': 2}
    feed_embed_feat_nums = {'read_comment': 4, 'like': 8, 'click_avatar': 8, 'forward': 8}
    train_params = {}
    train_params['epochs'] = epochs
    train_params['FEA_FEED_LISTs'] = FEA_FEED_LISTs
    train_params['FEA_ACTIONS'] = FEA_ACTIONS
    train_params['embedding_dims'] = embedding_dims
    train_params['dnn_hidden_units'] = dnn_hidden_units
    train_params['feed_embed_feat_nums'] = feed_embed_feat_nums

    if mode == 'eval':
        train_deepfm_eval(train_params)
    else:
        res = train_deepfm(train_params)
        return res


def bagging(L):
    for i in range(L):
        print("training predictor %d" % (i + 1))
        if os.path.exists("./dataset/test_data.csv"):
            print("train files already generated")
        else:
            prepare_datasets()
        train_datasets()
        # res.to_csv("./submit_advance_deepfm.csv", index=False)


if __name__ == '__main__':
    bagging(1)
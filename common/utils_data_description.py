from utils_load_data import *
import pandas as pd
import math


def describe_keyword_tag():
    feed_info = load_feed_info()
    manual_keyword_list = feed_info['manual_keyword_list'].to_list()
    machine_keyword_list = feed_info['machine_keyword_list'].to_list()
    manual_keyword_list = [list(map(int, l.split(';'))) if (isinstance(l, str) or not math.isnan(l)) else float('nan')
                           for l in manual_keyword_list]
    machine_keyword_list = [list(map(int, l.split(';'))) if (isinstance(l, str) or not math.isnan(l)) else float('nan')
                            for l in machine_keyword_list]

    # 统计不同关键词的数量和出现次数
    manual_keyword_words = {}
    for l in manual_keyword_list:
        if isinstance(l, float) and math.isnan(l): continue
        for keyword in l:
            try:
                manual_keyword_words[keyword] += 1
            except KeyError:
                manual_keyword_words[keyword] = 1

    # 统计不同关键词的数量和出现次数
    machine_keyword_words = {}
    for l in machine_keyword_list:
        if isinstance(l, float) and math.isnan(l): continue
        for keyword in l:
            try:
                machine_keyword_words[keyword] += 1
            except KeyError:
                machine_keyword_words[keyword] = 1
    print(len(manual_keyword_words))
    print(len(machine_keyword_words))

    manual_tag_list = feed_info['manual_tag_list'].to_list()
    manual_tag_list = [list(map(int, l.split(';'))) if (isinstance(l, str) or not math.isnan(l)) else float('nan')
                           for l in manual_tag_list]
    manual_tag_words = {}
    for l in manual_tag_list:
        if isinstance(l, float) and math.isnan(l): continue
        for keyword in l:
            try:
                manual_tag_words[keyword] += 1
            except KeyError:
                manual_tag_words[keyword] = 1
    print(len(manual_tag_words))


def describe_bmg_singer():
    feed_info = load_feed_info()
    bgm_song = feed_info['bgm_song_id'].to_list()
    bgm_singer = feed_info['bgm_singer_id'].to_list()
    bgm_song_set = set(bgm_song)
    bgm_singer_set = set(bgm_singer)
    print(len(bgm_song_set))
    print(len(bgm_singer_set))


def covariance():
    """
    分析用户行为相关性
    :return:
    """
    user_action = load_user_action()
    action = user_action[['read_comment', 'like', 'click_avatar', 'forward', 'play', 'stay']]
    pd.set_option('display.max_columns', None)
    print(action[:5])

    print(action.corr())


if __name__ == '__main__':
    describe_bmg_singer()
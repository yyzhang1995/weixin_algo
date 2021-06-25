import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import gc
from collections import defaultdict

__all__ = ['process_machine_tag_list', 'process_manual_tag_list', 'process_machine_keyword', 'process_manual_keyword',
           'process_embed', 'reduce_mem', 'uAUC']


def process_machine_tag_list(data):
    n = data.shape[0]
    machine_tag = np.zeros(n)
    print("processing machine_tag_list")
    for i in tqdm(range(n)):
        x = data.loc[i, 'machine_tag_list']
        try:
            machine_tag[i] = int(x.split(';')[0].split(' ')[0])
        except AttributeError:
            machine_tag[i] = -1
    data['machine_tag_list'] = machine_tag


def process_manual_tag_list(data):
    n = data.shape[0]
    num_tags = 352
    emergence_time = np.zeros(num_tags + 1)
    print("processing manual_tag_list")
    # # 先构建emergence_time表
    # for i in range(n):
    #     x = data.loc[i, 'manual_tag_list']
    #     try:
    #         tags = [int(xi) for xi in x.split(';')]
    #     except AttributeError:
    #         continue
    #     for tag in tags:
    #         emergence_time[tag] += 1
    # 构建manual_tag_list特征
    manual_tag = np.zeros(n)
    for i in tqdm(range(n)):
        x = data.loc[i, 'manual_tag_list']
        try:
            manual_tag[i] = int(x.split(';')[0])
        except AttributeError:
            manual_tag[i] = 0
            continue
    data['manual_tag_list'] = manual_tag


def process_machine_keyword(data):
    n = data.shape[0]
    machine_keyword = np.zeros(n)
    print("processing machine_keyword_list")
    for i in tqdm(range(n)):
        x = data.loc[i, 'machine_keyword_list']
        try:
            machine_keyword[i] = int(x.split(';')[0])
        except AttributeError:
            machine_keyword[i] = 0
            continue
    data['machine_keyword_list'] = machine_keyword


def process_manual_keyword(data):
    n = data.shape[0]
    manual_keyword = np.zeros(n)
    print("processing manual_keyword_list")
    for i in tqdm(range(n)):
        x = data.loc[i, 'manual_keyword_list']
        try:
            manual_keyword[i] = int(x.split(';')[0])
        except AttributeError:
            manual_keyword[i] = 0
    data['manual_keyword_list'] = manual_keyword


def process_embed(train):
    feed_embed_array = np.zeros((train.shape[0], 512))
    print("processing feed embeddings")
    for i in tqdm(range(train.shape[0])):
        x = train.loc[i, 'feed_embedding']
        if x != np.nan and x != '':
            y = [float(i) for i in str(x).strip().split(" ")]
        else:
            y = np.zeros((512,)).tolist()
        feed_embed_array[i] += y

    pca = PCA(n_components=32, whiten=True)
    feed_embed_pca = pca.fit_transform(feed_embed_array)
    temp = pd.DataFrame(columns=[f"embed{i}" for i in range(32)], data=feed_embed_pca)
    feed_embed_pca = pd.concat((train[['feedid']], temp), axis=1)
    return feed_embed_pca


def reduce_mem(df, cols):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in tqdm(cols):
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df


def uAUC(labels, preds, user_id_list):

    user_pred = defaultdict(lambda: [])
    user_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        pred = preds[idx]
        truth = labels[idx]
        user_pred[user_id].append(pred)
        user_truth[user_id].append(truth)

    user_flag = defaultdict(lambda: False)

    for user_id in set(user_id_list):
        truths = user_truth[user_id]
        flag = False
        # 若全是正样本或全是负样本，则flag为False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break

        user_flag[user_id] = flag
    total_auc = 0.0
    size = 0.0
    for user_id in user_flag:
        if user_flag[user_id]:
            auc = roc_auc_score(np.asarray(user_truth[user_id]), np.asarray(user_pred[user_id]))
            total_auc += auc
            size += 1.0
    user_auc = float(total_auc)/size
    return user_auc

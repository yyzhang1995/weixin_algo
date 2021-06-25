import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from collections import defaultdict

__all__ = ['process_machine_tag_list', 'process_manual_tag_list', 'process_machine_keyword', 'process_manual_keyword',
           'process_embed', 'uAUC']


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

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from deepctr_torch.layers.utils import slice_arrays

__all__ = ['process_machine_tag_list', 'process_manual_tag_list', 'process_machine_keyword', 'process_manual_keyword',
           'process_embed', 'shuffle_k_fold']


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


def concat_arrays(arrays):
    return [pd.concat(z) for z in zip(*arrays)]


def shuffle_k_fold(arrays, k):
    """
    将数据的最下方1/k挪到最上方
    :param arrays:
    :param k:
    :return:
    """
    validation_split = 1 / k

    if isinstance(arrays, dict):
        for feat in arrays.keys():
            if hasattr(arrays[feat], 'shape'):
                split_at = int(arrays[feat].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(arrays[feat]) * (1. - validation_split))
            arrays1, arrays2 = (slice_arrays([arrays[feat]], 0, split_at),
                                slice_arrays([arrays[feat]], split_at))
            arrays[feat] = pd.concat([arrays2, arrays1])
        return arrays

    if hasattr(arrays, 'shape'):
        arrays = [arrays]

    if hasattr(arrays[0], 'shape'):
        split_at = int(arrays[0].shape[0] * (1. - validation_split))
    else:
        split_at = int(len(arrays[0]) * (1. - validation_split))
    arrays1, arrays2 = (slice_arrays(arrays, 0, split_at),
                        slice_arrays(arrays, split_at))
    if len(arrays) == 1:
        return pd.concat([arrays2, arrays1])
    else:
        return concat_arrays([arrays2, arrays1])


if __name__ == '__main__':
    x1 = pd.Series([1,2,3,4,5])
    x2 = pd.Series([11, 12, 13, 14, 15])
    x3 = pd.Series([21, 22, 23, 24, 25])

    y1 = pd.Series([6,7,8,9,10])
    y2 = pd.Series([16, 17, 18, 19, 20])
    y3 = pd.Series([26, 27, 28, 29, 30])

    x = [x1, x2, x3]
    y = [y1, y2, y3]

    d = {}
    d['x1'] = x1
    d['x2'] = x2

    print(shuffle_k_fold(d, 5))
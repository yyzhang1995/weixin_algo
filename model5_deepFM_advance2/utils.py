import numpy as np
import pandas as pd
from tqdm import tqdm

__all__ = ['process_machine_tag_list', 'process_manual_tag_list', 'process_machine_keyword', 'process_manual_keyword']


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
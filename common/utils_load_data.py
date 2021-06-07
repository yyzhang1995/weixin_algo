import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch


class UserActionFeatures(Dataset):
    def __init__(self, action_user, feed_info_user, feed_embeddings_user, behavior):
        self.action_user = action_user
        self.feed_info_user = feed_info_user
        self.feed_embeddings_user = change_feed_embeddings_to_tensor(feed_embeddings_user)
        self.behavior = behavior

    def __len__(self):
        return self.action_user.shape[0]

    def __getitem__(self, index):
        feed_id = self.action_user['feedid'].iloc[index]
        device_id = self.action_user['device'].iloc[index]
        feed_features_wide = self.feed_info_user[self.feed_info_user['feedid'] == feed_id].to_numpy()
        feed_features_wide[0, 0] = device_id
        feed_features_wide = torch.tensor(feed_features_wide, dtype=torch.float32).view(feed_features_wide.shape[1])
        feed_features_deep = self.feed_embeddings_user[self.feed_embeddings_user[:, 0] == feed_id, :][:, 1:].squeeze(0)
        return feed_features_wide, feed_features_deep, self.action_user[self.behavior].iloc[index]


def change_feed_embeddings_to_tensor(df):
    feedid = df['feedid'].to_list()
    feed_embedding = df['feed_embedding'].to_list()
    feed_embedding = [list(map(float, l.strip().split(' '))) for l in feed_embedding]
    feedid = torch.tensor(feedid).view(-1, 1)
    feed_embedding = torch.tensor(feed_embedding)
    return torch.cat((feedid, feed_embedding), dim=1)


def load_feed_info():
    feed_info = '../dataset/feed_info.csv'
    f = pd.read_csv(feed_info)
    f = f.fillna(0)
    # print(f['manual_keyword_list'][:5])
    return f


def load_feed_embeddings():
    feed_embedding = '../dataset/feed_embeddings.csv'
    f = pd.read_csv(feed_embedding)
    # f = change_feed_embeddings_to_tensor(f)
    return f


def load_user_action():
    user_action = '../dataset/user_action.csv'
    f = pd.read_csv(user_action)
    return f


def query_feed_info_based_on_user_id(user_id, user_action, feed_info, feed_embeddings):
    feed_ids = user_action[user_action['userid'] == user_id]['feedid']
    return feed_info[feed_info['feedid'].isin(feed_ids)], feed_embeddings[feed_embeddings['feedid'].isin(feed_ids)]


def load_data_by_userid(user_id, batch_size, behavior, user_action=None, feed_info=None, feed_embeddings=None):
    if user_action is None:
        user_action = load_user_action()
    if feed_info is None:
        feed_info = load_feed_info()
    if feed_embeddings is None:
        feed_embeddings = load_feed_embeddings()
    feed_info_user, feed_embeddings_user = query_feed_info_based_on_user_id(user_id, user_action,
                                                                            feed_info, feed_embeddings)
    wide_features = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']
    feed_info_user = feed_info_user[wide_features]

    user_action_user = user_action[user_action['userid'] == user_id]
    # 进行train和test的分割，分割规则：1-13天为训练，14天为验证集数据
    user_action_train = user_action_user[user_action_user['date_'] <= 13]
    user_action_valid = user_action_user[user_action_user['date_'] == 14]
    if user_action_train.shape[0] == 0 or user_action_valid.shape[0] == 0:
        return None, None

    Dataset_train = UserActionFeatures(user_action_train, feed_info_user, feed_embeddings_user, behavior)
    Dataset_valid = UserActionFeatures(user_action_valid, feed_info_user, feed_embeddings_user, behavior)
    train_iter = DataLoader(Dataset_train, batch_size=batch_size, shuffle=False) # 不shuffle的理由是考虑到时间顺序问题
    valid_iter = DataLoader(Dataset_valid, batch_size=batch_size, shuffle=False)
    return train_iter, valid_iter


if __name__ == '__main__':
    pass
    # user_action = load_user_action()
    # feed_info = load_feed_info()
    # feed_info_user = query_feed_info_based_on_user_id(user_id=8, user_action=user_action, feed_info=feed_info)
    # behavior = 'like'
    # print(user_action[behavior].iloc[:20])

    train_iter, valid_iter = load_data_by_userid(user_id=8, batch_size=8, behavior='like')
    for X_wide, X_deep, y in train_iter:
        print(X_wide, X_deep, y)
        break

    # load_feed_embeddings()
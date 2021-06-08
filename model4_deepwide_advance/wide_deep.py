import torch
from torch import nn, optim
from common.utils_load_data import *
from common.utils_eval import *

from deepctr_torch.inputs import SparseFeat, DenseFeat
from sklearn.preprocessing import LabelEncoder

class WideDeepNet(nn.Module):
    def __init__(self, wide_input_dim, deepnet):
        super(WideDeepNet, self).__init__()
        self.deepnet = deepnet
        self.widenet = nn.Linear(in_features=wide_input_dim, out_features=1)

    def forward(self, wide_net_X, deepnet_X):
        wide_X = self.widenet(wide_net_X)
        deepnet_X = self.deepnet(deepnet_X)
        Y = wide_X + deepnet_X
        return Y


def deep(deep_input_dim):
    net = nn.Sequential(
        nn.Linear(in_features=deep_input_dim, out_features=256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(in_features=256, out_features=128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(in_features=128, out_features=1)
    )
    return net


# # 验证网络可以正确运行
# deepnet = deep(100)
# test_net = WideDeepNet(wide_input_dim=20, deepnet=deepnet)
# wide_X = torch.rand(3, 20)
# deep_X = torch.rand(3, 100)
# print(test_net)
# print(test_net(wide_X, deep_X).shape)

def train_user_behavior(user_id, batch_size, behavior,  user_action, feed_info, feed_embeddings,
                        lr, num_epochs, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

    train_iter, valid_iter = load_data_by_userid(user_id, batch_size, behavior, user_action, feed_info, feed_embeddings)
    if train_iter is None:
        return 'invalid'

    # 定义网络
    deepnet = deep(512)
    wide_input_dim = 5
    net = WideDeepNet(wide_input_dim, deepnet)

    # # 定义超参数
    # lr = 0.01
    # num_epochs = 10

    # 定义损失函数和优化器
    loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # 开始训练
    net = net.to(device)
    for epoch in range(num_epochs):
        train_loss_sum, train_AUC, n = 0.0, 0.0, 0
        for X_wide, X_deep, y in train_iter:
            X_wide = X_wide.float().to(device)
            X_deep = X_deep.float().to(device)
            y = y.to(device)
            y_hat = net(X_wide, X_deep)
            l = loss(y_hat.squeeze(1), y.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_loss_sum += l.cpu().item()
            train_AUC = evaluate_AUC(net, train_iter, device)
            n += y.shape[0]
        test_AUC = evaluate_AUC(net, valid_iter, device)
        if test_AUC == 'invalid' or test_AUC == 1.0:
            return test_AUC
        print("epoch: %d, train loss: %.4f, train acc: %.3f, test AUC: %.3f" %
              (epoch + 1, train_loss_sum / n, train_AUC if not isinstance(train_AUC, str) else 1.0, test_AUC))

    return test_AUC


def get_train_user_id(user_action):
    user_ids = set(user_action['userid'].to_list())
    user_ids = list(user_ids)
    user_ids.sort()
    return user_ids


def train_on_train_set():
    user_action = load_user_action()
    feed_info = load_feed_info()
    feed_embeddings = load_feed_embeddings()

    behaviors = ['read_comment', 'like', 'forward', 'click_avatar']
    behavior = 'like'

    user_ids = get_train_user_id(user_action)

    # 定义超参数
    batch_size = 256
    lr = 0.1
    num_epochs = 1

    effective_user = 0
    AUC_sum = 0.0
    for i, user_id in enumerate(user_ids):
        res = train_user_behavior(user_id, batch_size, behavior, user_action, feed_info, feed_embeddings,
                                  lr, num_epochs)
        if res == 'invalid':
            print('client %d, id: %d' % (i, user_id), 'invalid')
            continue
        effective_user += 1
        AUC_sum += res
        print('client %d, id: %d' % (i, user_id), "test AUC: %.4f, uAUC: %.3f" % (res, AUC_sum / effective_user))
    print("AUC = %.3" % (AUC_sum / effective_user))


if __name__ == '__main__':
    # train_on_train_set()
    sparse_features = ['authorid', 'bgm_song_id', 'bgm_singer_id']
    sparse_feed_info = load_feed_info()[sparse_features]

    print(sparse_feed_info[:5])

    for feat in sparse_features:
        lbe = LabelEncoder()
        sparse_feed_info[feat] = lbe.fit_transform(sparse_feed_info[feat])

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=sparse_feed_info[feat].nunique(), embedding_dim=4)
                              for i, feat in enumerate(sparse_features)]

    print(fixlen_feature_columns)

import torch
from torch import nn, optim
from utils_load_data import *
import time

# from deepctr_torch.inputs import SparseFeat, DenseFeat
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset, DataLoader

DATA_SOURCE_FILE = './preprocessing_data/'
ACTIONS = ['read_comment', 'like', 'click_avatar', 'forward']

SPARSE_FEATURES = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id']
DENSE_FEATURES = ['videoplayseconds']

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
        nn.Linear(in_features=deep_input_dim, out_features=512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(in_features=512, out_features=256),
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

def train_valid_split(train_data):
    # 选择第14天的训练数据作为验证集
    valid_data = train_data[train_data['date_'] == 14]
    train_data = train_data[train_data['date_'] <= 13]
    return train_data, valid_data


def feature_embedding(*data):
    # 对SparseFeature进行embedding处理
    is_list = False
    data_num = 1
    if isinstance(data, tuple) or isinstance(data, list):
        split_point = []
        length = 0
        data_num = len(data)
        for d in data:
            length += d.shape[0]
            split_point.append(length)
        data = pd.concat(data)
        is_list = True
    for sparse_feat in SPARSE_FEATURES:
        lbe = LabelEncoder()
        data[sparse_feat] = lbe.fit_transform(data[sparse_feat])
    sparse_feature_dict = {}
    for sparse_feat in SPARSE_FEATURES:
        nunique = data[sparse_feat].nunique()
        embedding = nn.Embedding(nunique, 4)
        feature_embedded = torch.from_numpy(data[sparse_feat].to_numpy())
        with torch.no_grad():
            feature_embedded = embedding(feature_embedded)
        sparse_feature_dict[sparse_feat] = feature_embedded
    # 对DenseFeature进行处理
    mms = MinMaxScaler(feature_range=(0, 1))
    data[DENSE_FEATURES] = mms.fit_transform(data[DENSE_FEATURES])
    # 最后进行特征整合
    dense_feature_number = len(DENSE_FEATURES)
    feature_all = torch.zeros(size=(split_point[-1], dense_feature_number))
    feature_point = 0
    for dense_feature in DENSE_FEATURES:
        feature_all[:, feature_point] = torch.from_numpy(data[dense_feature].to_numpy())
        feature_point += 1
    # 直接拼接sparse features
    for name in sparse_feature_dict.keys():
        feature_all = torch.cat((feature_all, sparse_feature_dict[name]), dim=1)
    # 特征切分
    if not is_list:
        return feature_all
    data_splited = []
    for i in range(data_num):
        if i == 0:
            data_splited.append(feature_all[0:split_point[i], :])
        else:
            data_splited.append(feature_all[split_point[i - 1] : split_point[i], :])
    return data_splited


def evaluate(net, iter, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    net.eval()
    y_pred, y_true = None, None
    for X, y in iter:
        with torch.no_grad():
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X, X).squeeze(1)
        if y_pred is None:
            y_pred = y_hat
            y_true = y
        else:
            y_pred = torch.cat((y_pred, y_hat))
            y_true = torch.cat((y_true, y))

    net.train()
    AUC = roc_auc_score(y_score=y_pred.cpu(), y_true=y_true.cpu())
    return AUC


def train_user_behavior(train_data, valid_data, train_label, valid_label, batch_size, num_epochs, lr=0.1,
                        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

    ds_train = TensorDataset(train_data, train_label)
    ds_valid = TensorDataset(valid_data, valid_label)
    train_iter = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    valid_iter = DataLoader(ds_valid, batch_size=batch_size, shuffle=False)

    # 定义网络
    deepnet = deep(21)
    wide_input_dim = 21
    net = WideDeepNet(wide_input_dim, deepnet)

    # # 定义超参数
    # lr = 0.01
    # num_epochs = 10

    # 定义损失函数和优化器
    loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # 开始训练
    print("train on", device)
    net = net.to(device)
    for epoch in range(num_epochs):
        begin = time.time()
        train_loss_sum, n = 0.0, 0
        for X, y in train_iter:
            optimizer.zero_grad()
            X = X.float().to(device)
            y = y.to(device)
            y_hat = net(X, X)
            l = loss(y_hat.squeeze(1), y.float())
            l.backward(retain_graph=True)
            optimizer.step()
            train_loss_sum += l.cpu().item()
            n += y.shape[0]
        train_AUC = evaluate(net, train_iter, device)
        test_AUC = evaluate(net, valid_iter, device)
        t = time.time() - begin
        print("time: %d, epoch: %d, train loss: %.5f, train AUC: %.4f, test AUC: %.4f" %
              (t, epoch + 1, train_loss_sum / n, train_AUC if not isinstance(train_AUC, str) else 1.0, test_AUC))
    return net


def train_and_validate():
    for action in ACTIONS:
        if action != 'like': # done : read_comment(20) forward(20) click_avatar(20)
            continue
        print("train on " + action)
        # 读取数据
        train_data = pd.read_csv(DATA_SOURCE_FILE + ('train_data_for_%s.csv' % action))
        test_data = pd.read_csv(DATA_SOURCE_FILE + 'test_data.csv')
        # 对训练数据划分训练集和测试集
        train_data, valid_data = train_valid_split(train_data)
        # 读取标签
        train_label = torch.from_numpy(train_data[action].to_numpy())
        valid_label = torch.from_numpy(valid_data[action].to_numpy())
        # 将输入的数据集进行连接以后，进行embedding处理, 并转化为tensor
        train_data, valid_data, test_data = feature_embedding(train_data, valid_data, test_data)

        # 设置超参数
        batch_size = 512
        lrs = {'read_comment': 1e-3, 'like': 1e-3, 'click_avatar': 1e-3, 'forward': 5e-4}
        num_epochs = 20
        # 训练
        net_trained = train_user_behavior(train_data, valid_data, train_label, valid_label,
                                          batch_size, num_epochs, lrs[action])


if __name__ == '__main__':
    # train_on_train_set()

    # sparse_features = ['authorid', 'bgm_song_id', 'bgm_singer_id']
    # sparse_feed_info = load_feed_info()[sparse_features]
    #
    # print(sparse_feed_info[:5])
    #
    # for feat in sparse_features:
    #     lbe = LabelEncoder()
    #     sparse_feed_info[feat] = lbe.fit_transform(sparse_feed_info[feat])
    #
    # fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=sparse_feed_info[feat].nunique(), embedding_dim=4)
    #                           for i, feat in enumerate(sparse_features)]
    #
    # print(fixlen_feature_columns)
    train_and_validate()

    # test_data = pd.read_csv(DATA_SOURCE_FILE + 'test_data.csv')
    # train_data = pd.read_csv(DATA_SOURCE_FILE + 'train_data_for_click_avatar.csv')
    # d = pd.concat((train_data, test_data))
    # print(train_data.shape)
    # print(d[1484127:1484132])
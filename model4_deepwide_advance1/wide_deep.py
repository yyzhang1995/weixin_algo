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


class DeepFMBase(nn.Module):
    def __init__(self, feature_size, dense_feature_num, embedding_size, hidden_nums):
        """
        前dense_feature_num个变量都属于dense变量, 其余变量为稀疏变量
        :param feature_size: 列表, 值为1表示dense变量,否则为稀疏变量的相异元素个数
        :param dense_feature_num:
        :param embedding_size:
        :param hidden_nums: 隐藏层神经元个数,可以是list或int
        """
        super(DeepFMBase, self).__init__()
        self.field_size = len(feature_size)
        self.dense_feature_num = dense_feature_num
        # 一阶项
        first_order_dense = nn.ModuleList([
            nn.Linear(1, embedding_size) for _ in range(dense_feature_num)
        ])
        first_order_sparse = nn.ModuleList([
            nn.Embedding(feature_size_sparse, embedding_size, max_norm=1e-1) for feature_size_sparse in
            feature_size[dense_feature_num:]
        ])
        self.first_order_embedding = first_order_dense.extend(first_order_sparse)
        self.first_order_linear = nn.Linear(self.field_size, 1)

        # 二阶项和深度学习部分
        second_order_dense = nn.ModuleList([
            nn.Linear(1, embedding_size) for _ in range(dense_feature_num)
        ])
        second_order_sparse = nn.ModuleList([
            nn.Embedding(feature_size_sparse, embedding_size, max_norm=1e-1) for feature_size_sparse in
            feature_size[dense_feature_num:]
        ])
        self.second_order_embedding = second_order_dense.extend(second_order_sparse)

        neuron_nums = [self.field_size * embedding_size] + \
                      (hidden_nums if isinstance(hidden_nums, list) else [hidden_nums]) + [1]
        self.deepnet = deep(neuron_nums)

    def forward(self, X):
        """

        :param X: X size:(batch_size, field_size)
        :return:
        """
        # 一阶embedding
        first_order_embedded = []
        for i, embedding_fun in enumerate(self.first_order_embedding):
            if i < self.dense_feature_num:
                Xi = X[:, i].float()
                first_order_embedded.append(torch.sum(embedding_fun(Xi.view(-1, 1)), dim=1).view(-1, 1))
            else:
                Xi = X[:, i].long()
                first_order_embedded.append(torch.sum(embedding_fun(Xi), dim=1).view(-1, 1))
        first_order_features = torch.cat(first_order_embedded, dim=1)
        first_order_output = self.first_order_linear(first_order_features)

        # 二阶embedding
        second_order_embedded = []
        for i, embedding_fun in enumerate(self.second_order_embedding):
            if i < self.dense_feature_num:
                Xi = X[:, [i]].float()
                second_order_embedded.append(embedding_fun(Xi))
            else:
                Xi = X[:, i].long()
                second_order_embedded.append(embedding_fun(Xi))
        second_order_features = torch.cat(second_order_embedded, dim=1)
        # 计算交叉项
        second_order_features_sum_square = (sum(second_order_features) ** 2)
        second_order_features_squared_sum = [feature ** 2 for feature in second_order_features]
        second_order_features_squared_sum = sum(second_order_features_squared_sum)
        second_order_output = 0.5 * torch.sum(second_order_features_sum_square - second_order_features_squared_sum)

        # deep部分
        deep_output = self.deepnet(second_order_features)

        # 结合项
        output = first_order_output + second_order_output + deep_output

        return output


def deep(neuron_nums, dropout_rates=0.5, batch_norm=False):
    """

    :param neuron_nums: list, 长度减1就是层数
    :param dropout_rates: list或int
    :param batch_norm: bool, 是否需要添加batch normalization层
    :return:
    """
    linear_nums = len(neuron_nums) - 1
    net = nn.Sequential(nn.Linear(neuron_nums[0], neuron_nums[1]))
    for i in range(1, linear_nums):
        module = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout_rates if isinstance(dropout_rates, float) else dropout_rates[i]),
            nn.Linear(neuron_nums[i], neuron_nums[i + 1])
        )
        net.add_module(name='deep_linear_%d' % (i + 1), module=module)
    return net


def train_valid_split(train_data):
    # 选择第14天的训练数据作为验证集
    valid_data = train_data[train_data['date_'] == 14]
    train_data = train_data[train_data['date_'] <= 13]
    return train_data, valid_data


def get_feature_size(*data):
    if isinstance(data, tuple) or isinstance(data, list):
        data = pd.concat(data)
    dense_feature_size = [1] * len(DENSE_FEATURES)
    sparse_feature_size = data[SPARSE_FEATURES].nunique().to_list()
    return dense_feature_size + sparse_feature_size


def feature_extraction(*data):
    # 根据设定的特征进行提取
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
    # 对DenseFeature进行处理
    mms = MinMaxScaler(feature_range=(0, 1))
    data[DENSE_FEATURES] = mms.fit_transform(data[DENSE_FEATURES])

    # 提取特征
    data = data[DENSE_FEATURES + SPARSE_FEATURES]
    feature_all = torch.from_numpy(data.to_numpy())

    # 特征切分
    if not is_list:
        return feature_all
    data_splited = []
    for i in range(data_num):
        if i == 0:
            data_splited.append(feature_all[0:split_point[i], :])
        else:
            data_splited.append(feature_all[split_point[i - 1]: split_point[i], :])
    return data_splited


def train_user_behavior(train_data, valid_data, train_label, valid_label, feature_size, batch_size,
                        num_epochs, lr=0.1, embedding_size=4, hidden_nums=[128, 64],
                        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    ds_train = TensorDataset(train_data, train_label)
    train_iter = DataLoader(ds_train, batch_size=batch_size, shuffle=True)

    if valid_data is not None:
        ds_valid = TensorDataset(valid_data, valid_label)
        valid_iter = DataLoader(ds_valid, batch_size=batch_size, shuffle=False)

    # 定义网络
    net = DeepFMBase(feature_size=feature_size, dense_feature_num=len(DENSE_FEATURES),
                     embedding_size=embedding_size, hidden_nums=hidden_nums)

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
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat.squeeze(1), y.float())
            l.backward(retain_graph=True)
            optimizer.step()
            train_loss_sum += l.cpu().item()
            n += y.shape[0]
        train_AUC = evaluate(net, train_iter, device)
        if valid_data is not None:
            test_AUC = evaluate(net, valid_iter, device)
        else:
            test_AUC = -1
        t = time.time() - begin
        print("time: %d, epoch: %d, train loss: %.5f, train AUC: %.4f, test AUC: %.4f" %
              (t, epoch + 1, train_loss_sum / n, train_AUC, test_AUC))
    return net


def predict(net, test_data, batch_size, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    ds_test = TensorDataset(test_data)
    test_iter = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    net.eval()
    y_pred = None
    for X, in test_iter:
        with torch.no_grad():
            X = X.to(device)
            y_hat = net(X).squeeze(1)
        if y_pred is None:
            y_pred = y_hat
        else:
            y_pred = torch.cat((y_pred, y_hat))

    net.train()
    y_pred = nn.Sigmoid()(y_pred)
    return y_pred


def evaluate(net, iter, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    net.eval()
    y_pred, y_true = None, None
    for X, y in iter:
        with torch.no_grad():
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X).squeeze(1)
        if y_pred is None:
            y_pred = y_hat
            y_true = y
        else:
            y_pred = torch.cat((y_pred, y_hat))
            y_true = torch.cat((y_true, y))

    net.train()
    y_pred = nn.Sigmoid()(y_pred)
    print(y_pred[:10])
    print(y_true[:10])
    AUC = roc_auc_score(y_score=y_pred.cpu(), y_true=y_true.cpu())
    return AUC


def train_and_validate():
    for action in ACTIONS:
        if action != 'like':  # done : read_comment, like
            continue
        print("train action: " + action)
        # 读取数据
        train_data = pd.read_csv(DATA_SOURCE_FILE + ('train_data_for_%s.csv' % action))
        test_data = pd.read_csv(DATA_SOURCE_FILE + 'test_data.csv')
        # 对训练数据划分训练集和测试集
        train_data, valid_data = train_valid_split(train_data)
        # 读取标签
        train_label = torch.from_numpy(train_data[action].to_numpy())
        valid_label = torch.from_numpy(valid_data[action].to_numpy())
        # 提取feature_sizes
        feature_sizes = get_feature_size(train_data, valid_data, test_data)
        # 将输入的数据集进行连接以后，进行embedding处理, 并转化为tensor
        train_data, valid_data, test_data = feature_extraction(train_data, valid_data, test_data)

        # 设置超参数
        batch_size = 512
        lrs = {'read_comment': 1e-2, 'like': 1e-2, 'click_avatar': 1e-2, 'forward': 1e-2}
        num_epochs = 5
        embedding_size = 4
        num_hiddens = [128, 64]
        # 训练
        train_user_behavior(train_data, valid_data, train_label, valid_label, feature_sizes, batch_size,
                            num_epochs, lrs[action], embedding_size, num_hiddens)


def train_and_predict():
    test_output = pd.read_csv(DATA_SOURCE_FILE + 'test_data.csv')[['userid', 'feedid']]
    for action in ACTIONS:
        print("train action: " + action)
        # 读取数据
        train_data = pd.read_csv(DATA_SOURCE_FILE + ('train_data_for_%s.csv' % action))
        test_data = pd.read_csv(DATA_SOURCE_FILE + 'test_data.csv')
        # 读取标签
        train_label = torch.from_numpy(train_data[action].to_numpy())
        # 提取feature_sizes
        feature_sizes = get_feature_size(train_data, test_data)
        # 将输入的数据集进行连接以后，进行embedding处理, 并转化为tensor
        train_data, test_data = feature_extraction(train_data, test_data)

        # 设置超参数
        batch_size = 512
        lr = {'read_comment': 1e-2, 'like': 1e-2, 'click_avatar': 1e-2, 'forward': 1e-2}
        num_epochs = {'read_comment': 5, 'like': 5, 'click_avatar': 5, 'forward': 5}
        embedding_size = 4
        num_hiddens = [128, 64]
        # 训练
        net_trained = train_user_behavior(train_data, None, train_label, None, feature_sizes, batch_size,
                                          num_epochs[action], lr[action], embedding_size, num_hiddens)

        # 预测
        print("prediction start")
        y_pred = predict(net_trained, test_data, batch_size)
        test_output[action] = y_pred.cpu().numpy().astype(np.float)

    OUTPUT_FILE = DATA_SOURCE_FILE + 'submit_deepFMbase.csv'
    test_output.to_csv(OUTPUT_FILE, index=False)


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
    # train_and_predict()


def testnet():
    net = DeepFMBase([1, 10, 10], 1, embedding_size=4, hidden_nums=[64, 32])
    X = torch.tensor([[0.5, 3, 7],
                      [0.8, 1, 9],
                      [0.2, 2, 4],
                      [0.5, 3, 5]])
    print(net)
    print(net(X).shape)


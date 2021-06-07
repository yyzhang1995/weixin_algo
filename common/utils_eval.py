import torch
import matplotlib.pyplot as plt


def user_behavior_AUC(pred_prob, user_behavior):
    """

    :param pred_prob:
    :param user_behavior:
    :return:
    """
    n = pred_prob.shape[0]
    combined = torch.cat((pred_prob.view(-1, 1), user_behavior.view(-1, 1)), dim=1)
    combined = combined[(-combined[:, 0]).argsort(), :]
    total_positive = user_behavior.sum().item()
    total_negative = n - total_positive
    tp_num = 0
    fp_num = 0
    if total_positive == 0 or total_negative == 0:
        return 'invalid'
    points = torch.zeros(size=(n + 1, 2))
    for i in range(n):
        if combined[i, 1] == 1:
            tp_num += 1
        else:
            fp_num += 1
        points[i + 1, :] = torch.tensor([tp_num/total_positive, fp_num/total_negative])
    # plt.plot(points[:, 1], points[:, 0])
    # plt.show()
    # 计算AUC
    AUC = 0
    for i in range(n):
        AUC += (points[i + 1, 1] - points[i, 1]) * (points[i + 1, 0] + points[i, 0]) / 2
    return AUC.item()


def AUC(pred_behaviors, user_behaviors):
    """

    :param pred_behaviors: list, 每个客户的预测结果
    :param user_behaviors: list, 每个客户的实际行为
    :return:
    """
    n = len(pred_behaviors)
    effective_n = 0
    AUC_read_comment, AUC_like, AUC_click_avatar, AUC_forward = 0.0, 0.0, 0.0, 0.0
    for i in range(n):
        # 判断该用户是否为正用户或者是负用户
        effective_n += 1
        pred_behavior = pred_behaviors[i]
        user_behavior = user_behaviors[i]
        AUC_read_comment += user_behavior_AUC(pred_behavior[:, 0], user_behavior[: 0])
        AUC_like += user_behavior_AUC(pred_behavior[:, 1], user_behavior[: 1])
        AUC_click_avatar += user_behavior_AUC(pred_behavior[:, 2], user_behavior[: 2])
        AUC_forward += user_behavior_AUC(pred_behavior[:, 3], user_behavior[: 3])

    AUC_read_comment, AUC_like, AUC_click_avatar, AUC_forward = AUC_read_comment / effective_n,\
    AUC_like / effective_n, AUC_click_avatar / effective_n, AUC_forward / effective_n
    # weights: read comment : 4, like : 3, click avatar : 2, forward : 1
    weighted_AUC = 4 * AUC_read_comment + 3 * AUC_like + 2 * AUC_click_avatar + AUC_forward
    return weighted_AUC


def evaluate_AUC(net, iter, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    net.eval()
    y_pred, y_true = None, None
    for X_wide, X_deep, y in iter:
        X_wide = X_wide.to(device)
        X_deep = X_deep.to(device)
        y = y.to(device)
        y_hat = net(X_wide, X_deep).squeeze(1)
        if y_pred is None:
            y_pred = y_hat
            y_true = y
        else:
            y_pred = torch.cat((y_pred, y_hat))
            y_true = torch.cat((y_true, y))
        AUC = user_behavior_AUC(y_pred, y_true)
    net.train()
    return AUC

if __name__ == '__main__':
    pred_prob = torch.tensor([0.7, 0.2, 0.3, 0.4, 0.5, 0.6, 0.55])
    user_behavior = torch.tensor([1, 0, 0, 1, 1, 1, 0])
    print(user_behavior_AUC(pred_prob, user_behavior))

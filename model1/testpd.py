import pandas as pd
import torch

#
# f = pd.DataFrame([[3, 1.1, 1],
#                   [8, 1.4, 1],
#                   [12, 9.0, 2],
#                   [18, -2.1, 3],
#                   [24, 14.1, 3],
#                   [29, 11.0, 5]], columns=('a', 'b', 'c'))
# b = pd.DataFrame([[12],
#                   [24]])
# print(f[['a', 'c']])

c = torch.tensor([[1, 0],
                  [2, -12],
                  [3, 13],
                  [4, 18],
                  [22, 19],
                  [23, 30]])

d = torch.tensor([1, 3, 22])

print(c[:, 0] == d)
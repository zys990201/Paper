import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix


def matrix(all_sessions, num_node):
    indptr, indices, data = [], [], []
    indptr.append(0)

    for j in range(len(all_sessions)):
        session = np.unique(all_sessions[j])
        length = len(session)
        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            indices.append(session[i]-1)
            data.append(1)
    matrix = csr_matrix((data, indices, indptr), shape=(len(all_sessions), num_node))

    return matrix


# 划分训练集与验证集
def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set # 样本，标签
    n_samples = len(train_set_x)         # 样本个数
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)

# 对于输入数据（train_data, valid_data, test_data)，进行处理
def handle_data(inputData, train_len=None):

    # 求出会话的最大长度max_len
    len_data = [len(nowData) for nowData in inputData]

    if train_len is None:
        max_len = max(len_data)
    else:
        max_len = train_len

    sess_len = []
    for session in inputData:
        nonzero_elems = np.nonzero(session)[0]
        sess_len.append([len(nonzero_elems)])

    sess_item = [list(upois) + [0] * (max_len - le) if le < max_len else list(upois[-max_len:])
                 for upois, le in zip(inputData, len_data)]
    reversed_sess_item = [list(reversed(upois)) + [0] * (max_len - le) if le < max_len else list(reversed(upois[-max_len:]))
               for upois, le in zip(inputData, len_data)]
    mask = [[1] * le + [0] * (max_len - le) if le < max_len else [1] * max_len
               for le in len_data]

    return sess_item, reversed_sess_item, mask, max_len, sess_len


class Data(Dataset):
    def __init__(self, data, num_node, train_len=None):
        sess_item, reversed_sess_item, mask, max_len, sess_len = handle_data(data[0], train_len)
        self.sess_item = np.asarray(sess_len)
        self.inputs = np.asarray(reversed_sess_item)
        self.raw = np.asarray(data[0])
        self.targets = np.asarray(data[1])
        self.mask = np.asarray(mask)
        self.length = len(data[0])
        self.max_len = max_len
        self.sess_len = np.asarray(sess_len)

        H_T = matrix(self.raw, num_node)
        BH_T = H_T.T.multiply(1.0 / H_T.sum(axis=1).reshape(1, -1))
        BH_T = BH_T.T
        H = H_T.T

        DH = H.T.multiply(1.0 / H.sum(axis=1).reshape(1, -1))
        DH = DH.T
        DHBH_T = np.dot(DH, BH_T)  # 40727 * 40727

        self.adjacency = DHBH_T.tocoo()
        self.n_node = num_node

    def __getitem__(self, index):
        sess_item, u_input, mask, target, sess_len = self.sess_item[index], self.inputs[index], self.mask[index], self.targets[index], self.sess_len[index]

        max_n_node = self.max_len
        node = np.unique(u_input)
        items = node.tolist() + (max_n_node - len(node)) * [0]

        adj = np.zeros((max_n_node, max_n_node))
        for i in np.arange(len(u_input) - 1):
            u = np.where(node == u_input[i])[0][0]
            adj[u][u] = 1

            if u_input[i + 1] == 0:
                break

            v = np.where(node == u_input[i + 1])[0][0]
            if u == v or adj[u][v] == 4:
                continue
            adj[v][v] = 1

            if adj[v][u] == 2:
                adj[u][v] = 4
                adj[v][u] = 4

            else:
                adj[u][v] = 2
                adj[v][u] = 3

        alias_inputs = [np.where(node == i)[0][0] for i in u_input]

        return [torch.tensor(sess_item), torch.tensor(alias_inputs), torch.tensor(adj), torch.tensor(items),
                torch.tensor(mask), torch.tensor(target), torch.tensor(u_input), torch.tensor(sess_len)]

    def __len__(self):
        return self.length

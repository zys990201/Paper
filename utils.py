import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix, coo_matrix

"""
# 构造邻接矩阵
def create_matrix(all_sessions, n_node):
    indptr, indices, data = [], [], [] # 索引指针(判断一行有几个元素), 索引(在行中所处位置), 数据 初始化为空列表
    indptr.append(0)

    # 遍历所有session
    for j in range(len(all_sessions)):
        session = np.unique(all_sessions[j]) # 去重session中的重复物品
        length = len(session) # 当前session长度
        s = indptr[-1] # 取索引指针的最后一个
        indptr.append((s + length))
        for i in range(length):
            indices.append(session[i]-1) # 行位置
            data.append(1) # 邻居为1
    matrix = csr_matrix((data, indices, indptr), shape=(len(all_sessions), n_node)) # 构造压缩稀疏行格式的稀疏矩

    return matrix
"""

def data_masks(all_sessions, n_node):
    adj = dict()
    # 遍历每一个会话，找到每个item的所有邻居以及对应出现次数
    for sess in all_sessions:
        for i, item in enumerate(sess):
            if i == len(sess)-1:
                break
            else:
                if sess[i] - 1 not in adj.keys():
                    adj[sess[i]-1] = dict()
                    adj[sess[i]-1][sess[i]-1] = 1  # I
                    adj[sess[i]-1][sess[i+1]-1] = 1
                else:
                    if sess[i+1]-1 not in adj[sess[i]-1].keys():
                        adj[sess[i] - 1][sess[i + 1] - 1] = 1
                    else:
                        adj[sess[i]-1][sess[i+1]-1] += 1
    row, col, data = [], [], []
    # 遍历每个物品，将其邻居以及其出现次数生成邻接矩阵（已添加I）
    for i in adj.keys():
        item = adj[i]
        for j in item.keys():
            row.append(i)
            col.append(j)
            data.append(adj[i][j])
    coo = coo_matrix((data, (row, col)), shape=(n_node, n_node))
    return coo


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

    # 返回反转并补0后的会话序列；us_msks是mask，有点击的为1，补0的为0；
    return reversed_sess_item, mask, max_len, sess_item, sess_len



# 继承Dataset类
class Data(Dataset):
    def __init__(self, data, n_node, all_train, train_len=None):
        reversed_sess_item, mask, max_len, sess_item, sess_len = handle_data(data[0], train_len)

        self.item_list = np.asarray(data[0])
        adj = data_masks(all_train, n_node)
        # # print(adj.sum(axis=0))
        self.adjacency = adj.multiply(1.0/adj.sum(axis=0).reshape(1, -1))
        """
        H_T = create_matrix(self.item_list, n_node)
        BH_T = H_T.T.multiply(1.0/H_T.sum(axis=1).reshape(1, -1))
        BH_T = BH_T.T
        H = H_T.T
        DH = H.T.multiply(1.0/H.sum(axis=1).reshape(1, -1))
        DH = DH.T
        DHBH_T = np.dot(DH,BH_T)

        self.adjacency = DHBH_T.tocoo()
        """
        self.sess_item = np.asarray(sess_item)
        self.sess_len = np.asarray(sess_len)
        self.reversed_sess_item = np.asarray(reversed_sess_item)
        self.mask = np.asarray(mask)
        self.targets = np.asarray(data[1])
        self.length = len(data[0])
        self.max_len = max_len
        self.n_node = n_node

    def __getitem__(self, index):
        # 根据索引index的值取出对应的会话序列u_input, mask, 以及target
        u_input, mask, target, sess_item, sess_len = self.reversed_sess_item[index], self.mask[index], self.targets[index], self.sess_item[index], self.sess_len[index]

        # max_n_node存储会话的最大长度，node是对输入会话进行去重，items对node进行补0
        max_n_node = self.max_len
        node = np.unique(u_input)
        items = node.tolist() + (max_n_node - len(node)) * [0] # list69

        # 构建邻接矩阵
        adj = np.zeros((max_n_node, max_n_node)) # 69 * 69
        for i in np.arange(len(u_input) - 1):
            u = np.where(node == u_input[i])[0][0] # 找到当前item在node列表中的位置，用u表示
            adj[u][u] = 1 # 添加self-loop

            # 如果下一个item是0，则退出
            if u_input[i + 1] == 0:
                break

            # 找到i的后一项u_input[i + 1]在会话中的位置，用v表示
            v = np.where(node == u_input[i + 1])[0][0]
            if u == v or adj[u][v] == 4: # 如果相邻两项为同一个item或有双向边
                continue
            adj[v][v] = 1 # 为后一项添加自循环

            # 如果v有到u的边，由于现在u到v也有边，因此u和v之间具有双向的边，双向的边设为4
            if adj[v][u] == 2:
                adj[u][v] = 4
                adj[v][u] = 4

            # 如果v到u没边，即只有u到v之间的边，因此adj[u][v]=2(表示u的出边）, adj[v][u]=3（表示v的入边）
            else:
                adj[u][v] = 2
                adj[v][u] = 3

        # 记录会话序列中所有item在node中对应的位置
        alias_inputs = [np.where(node == i)[0][0] for i in u_input]

        return [torch.tensor(alias_inputs), torch.tensor(adj), torch.tensor(items), torch.tensor(mask),
                torch.tensor(target), torch.tensor(u_input), torch.tensor(sess_item), torch.tensor(sess_len)]

    def __len__(self):
        return self.length

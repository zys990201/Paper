import datetime
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.nn import Module, Parameter
import torch.nn.functional as F


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


class Inter_S_view(nn.Module):
    def __init__(self, layers, dataset, emb_size=100):  # 初始化：层数，数据集，嵌入维度
        super(Inter_S_view, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
        self.dataset = dataset

    # 正向传播 input：邻居，Embedding  output：物品Embedding
    def forward(self, adjacency, embedding):
        item_embeddings = embedding
        item_embedding_layer0 = item_embeddings # 初始（第0层）embedding
        final = [item_embedding_layer0]
        # 经过i层卷积，将每层的卷积结果保存至final列表
        for i in range(self.layers):
            item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), item_embeddings)  # sparse.mm做矩阵乘法计算, item_embedding = 邻接矩阵 * 前一层embedding
            final.append(item_embeddings) # 将每层结果均添加到final数组中
     #  final1 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in final]))
     #  item_embeddings = torch.sum(final1, 0)
        item_embeddings = np.sum(final, 0) / (self.layers+1) # 最终物品Embedding的结果为之前所有计算结果取平均
        return item_embeddings # Xh


# 从model.py中知道这里forword函数的输入分别是会话中所有item的embedding：hidden， item的邻居item矩阵：adj，掩码：mask_item
# session-level graph 信息聚合
class Intra_S_view(nn.Module):
    def __init__(self, dim, alpha, dropout=0., name=None):
        super(Intra_S_view, self).__init__()
        self.dim = dim
        self.dropout = dropout
        # 可学习参数，4种权重向量
        self.a_0 = nn.Parameter(torch.Tensor(self.dim, 1)) # 100 * 1
        self.a_1 = nn.Parameter(torch.Tensor(self.dim, 1)) # 100 * 1
        self.a_2 = nn.Parameter(torch.Tensor(self.dim, 1)) # 100 * 1
        self.a_3 = nn.Parameter(torch.Tensor(self.dim, 1)) # 100 * 1
        self.bias = nn.Parameter(torch.Tensor(self.dim))   # 100 * 1

        self.leakyrelu = nn.LeakyReLU(alpha)

    # 输入：initial item embeedding，边类型矩阵
    # 输出：
    def forward(self, hidden, adj, mask_item=None):
        h = hidden
        batch_size = h.shape[0]
        N = h.shape[1]

        # 这里是公式(7)中点积部分：h_{v_i} * h_{v_j}
        a_input = (h.repeat(1, 1, N).view(batch_size, N * N, self.dim) * h.repeat(1, N, 1)).view(batch_size, N, N, self.dim)

        # 对a_input进行四种不同的线性映射表示四种类型的边
        e_0 = torch.matmul(a_input, self.a_0)
        e_1 = torch.matmul(a_input, self.a_1)
        e_2 = torch.matmul(a_input, self.a_2)
        e_3 = torch.matmul(a_input, self.a_3)

        # 进行LeakyReLU激活，公式(7)，得到注意力分数
        e_0 = self.leakyrelu(e_0).squeeze(-1).view(batch_size, N, N)
        e_1 = self.leakyrelu(e_1).squeeze(-1).view(batch_size, N, N)
        e_2 = self.leakyrelu(e_2).squeeze(-1).view(batch_size, N, N)
        e_3 = self.leakyrelu(e_3).squeeze(-1).view(batch_size, N, N)

        # 创建一个值全为-9e15的矩阵，如果adj中的值为1，则让alpha中对应的值为e_0中对应的值，否则alpha中对应的值为-9e15
        mask = -9e15 * torch.ones_like(e_0)
        alpha = torch.where(adj.eq(1), e_0, mask)
        # 这样alpha即由e_0和-9e15两个值组成，若其中的值等于2，则alpha中对应的值为e_1中对应的值，否则保持不变，3和4同理
        alpha = torch.where(adj.eq(2), e_1, alpha)
        alpha = torch.where(adj.eq(3), e_2, alpha)
        alpha = torch.where(adj.eq(4), e_3, alpha)
        alpha = torch.softmax(alpha, dim=-1)  # 对alpha进行softmax操作

        # 公式(9)加权求和,得到session-level item embedding
        output = torch.matmul(alpha, h)
        return output


class Session_view(Module):
    def __init__(self, layers, batch_size, emb_size=100):
        super(Session_view, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.layers = layers

    # input(item_embedding, D（线图度矩阵）, A（线图的邻接矩阵）, 会话物品, 会话长度) output(经线图卷积过后的session_embedding)
    def forward(self, item_embedding, D, A, session_item, session_len):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)  # 0矩阵（1*100）
        # zeros = torch.zeros([1,self.emb_size])
        item_embedding = torch.cat([zeros, item_embedding], 0) # 40727*100  # item_embedding矩阵
        seq_h = [] # 会话中的物品列表
        # 先通过查找属于每个会话的物品来初始化通道特定的会话嵌入
        for i in torch.arange(len(session_item)):
            # torch.index_select函数返回的是沿着输入张量的指定维度的指定索引号进行索引的张量子集
            # 返回item_embedding的第一个维度（行）上且索引号为第i个会话物品的子集
            seq_h.append(torch.index_select(item_embedding, 0, session_item[i]))
        # 返回一个新的tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置
        # 不同之处只是requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad
        # 如果想把CUDA tensor格式的数据改成numpy时，需要先将其转换成cpu float-tensor随后再转到numpy格式。
        # numpy不能读取CUDA tensor 需要将它转化为 CPU tensor
        seq_h1 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in seq_h]))
        session_emb_lgcn = torch.div(torch.sum(seq_h1, 1), session_len) # 初始化session embedding 由每个会话中的item embedding取平均得到
        session = [session_emb_lgcn]
        DA = torch.mm(D, A).float() # 度矩阵 * 邻接矩阵
        # 经过i层卷积，将每层的卷积结果保存至session列表
        for i in range(self.layers):
            session_emb_lgcn = torch.mm(DA, session_emb_lgcn) # 公式（8）
            session.append(session_emb_lgcn)
        #session1 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in session]))
        #session_emb_lgcn = torch.sum(session1, 0)
        session_emb_lgcn = np.sum(session, 0) / (self.layers+1)  # session层信息传播结果为每层传播结果求平均
        return session_emb_lgcn # Θl


# 模型实现核心代码
class CombineGraph(Module):
    def __init__(self, opt, adjacency, num_node, dataset):
        super(CombineGraph, self).__init__()
        self.opt = opt

        self.batch_size = opt.batch_size         # 批量大小=100
        self.num_node = num_node                 # 节点个数
        self.dim = opt.hiddenSize                # 隐藏层维度=100
        self.dropout_local = opt.dropout_local   # Dropout rate
        self.dropout_global = opt.dropout_global # Global_graph Dropout rate
        self.dropout = nn.Dropout(0.2)
        self.layers = opt.layer
        self.dataset = dataset

        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))

        """
        if dataset == 'Nowplaying':
            index_fliter = (values < 0.05).nonzero()  # 返回values<0.05的元素的索引值
            values = np.delete(values, index_fliter)
            indices1 = np.delete(indices[0], index_fliter)
            indices2 = np.delete(indices[1], index_fliter)
            indices = [indices1, indices2]
        """

        i = torch.LongTensor(indices) # 邻接矩阵行列索引
        v = torch.FloatTensor(values) # 邻接矩阵值
        shape = adjacency.shape
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))

        self.adjacency = adjacency

        # View
        self.Inter_S_view = Inter_S_view(self.layers, dataset)
        self.Intra_S_view = Intra_S_view(self.dim, self.opt.alpha, dropout=0.0)
        self.Session_view = Session_view(self.layers, self.batch_size)

        # Item representation & Position representation 初始化
        self.embedding = nn.Embedding(num_node, self.dim) # item_embedding 43098 * 100
        self.pos_embedding = nn.Embedding(200, self.dim)  # position_embedding 200 * 100

        # Parameters
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim)) # 200 * 100
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))            # 100 * 1

        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)
        self.linear_transform = nn.Linear(self.dim, self.dim, bias=False)

        self.leakyrelu = nn.LeakyReLU(opt.alpha) # 激活函数：LeakyRulu
        self.loss_function = nn.CrossEntropyLoss() # 损失函数：交叉熵损失函数
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2) # 优化器
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc) # 学习率调整策略：等间隔调整

        self.reset_parameters()

    # 从均匀分布中抽样数值进行初始化权重(-0.1,0.1)
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    # 得到最后的item表示并取出堆叠成会话，hidden的维度为：会话数*item数*embedding维度。
    def compute_scores(self, hidden, mask):
        mask = mask.float().unsqueeze(-1)

        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        # h_s即为s',公式12
        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)

        # 公式（11）将hidden与pos_emb拼接，通过tanh激活，n_h即为z_i
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        #nh = hidden
        nh = torch.tanh(nh)

        # 公式（13），z_i + s' = nh + hs, self.glu是线性映射层，得到beta
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask

        # 公式14，加权求和得到select = S.
        select = torch.sum(beta * hidden, 1)

        # 公式15，计算出\hat{y}=scores
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        b = self.dropout(b)
        b = F.normalize(b, dim=-1)
        scores = torch.matmul(select, b.transpose(1, 0))

        return select, scores

    # 输入会话，边类型矩阵，mask，会话中不重复的item列表（已补0）
    def forward(self, inputs, edge_matrix, mask, reversed_sess_item, sess_item, D, A, sess_len):
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]
        h = self.embedding(inputs) # initial item embedding

        intra_item_emb = self.Intra_S_view(h, edge_matrix, mask)
        inter_item_emb = self.Inter_S_view(self.adjacency, self.embedding.weight)
        session_emb = self.Session_view(self.embedding.weight, D, A, sess_item, sess_len)

        iter_item_emb = F.normalize(inter_item_emb, dim=-1, p=2)

        zeros = torch.cuda.FloatTensor(1, self.dim).fill_(0)
        # zeros = torch.zeros(1, self.emb_size)
        inter_item_emb = torch.cat([zeros, inter_item_emb], 0)
        get = lambda i: inter_item_emb[sess_item[i]]
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(sess_item.shape)[1], self.dim).fill_(0)
        # seq_h = torch.zeros(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size)
        for i in torch.arange(sess_item.shape[0]):
            seq_h[i] = get(i)

        # combine
        intra_item_emb = F.dropout(intra_item_emb, self.dropout_local, training=self.training)
        seq_h = F.dropout(seq_h, self.dropout_global, training=self.training)
        output =  intra_item_emb + seq_h

        return output, session_emb

    def SSL(self, sess_emb_hgnn, sess_emb_lgcn):
        # 随机行变换
        def row_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])] # 把行随机打乱生成新数组
            return corrupted_embedding
        # 随机行列变换
        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])] # 把行随机打乱
            corrupted_embedding = corrupted_embedding[:,torch.randperm(corrupted_embedding.size()[1])] # 把列随机打乱
            return corrupted_embedding # 返回负样本
        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 1) # 判别函数，对两个embedding进行一致性分析，采用点积

        pos = score(sess_emb_hgnn, sess_emb_lgcn) # 超图卷积生成的session embedding * 线图卷积生成的session embedding 公式（9）前半部分
        neg1 = score(sess_emb_lgcn, row_column_shuffle(sess_emb_hgnn)) # 线图卷积生成的embedding * 行列变换得到的负样本
        one = torch.cuda.FloatTensor(neg1.shape[0]).fill_(1)
        # one = zeros = torch.ones(neg1.shape[0])
        con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos))-torch.log(1e-8 + (one - torch.sigmoid(neg1)))) # 损失函数 公式（9）

        return con_loss # 返回联合损失


def forward(model, data):
    """
    alias_inputs:记录会话序列中所有item在node中对应的位置（node是对输入会话进行去重）
    adj:边类型矩阵
    items:转置的session序列
    mask:掩码序列
    targets:目标节点
    inputs:每个session物品列表
    """
    alias_inputs, edge_matrix, items, mask, targets, reversed_sess_item, sess_item, sess_len = data
    A, D = get_overlap(sess_item.tolist())

    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()
    edge_matrix = trans_to_cuda(edge_matrix).float()
    mask = trans_to_cuda(mask).long()
    reversed_sess_item = trans_to_cuda(reversed_sess_item).long()
    sess_item = trans_to_cuda(sess_item).long()
    sess_len = trans_to_cuda(sess_len).long()
    A = trans_to_cuda(torch.Tensor(A)).float()
    D = trans_to_cuda(torch.Tensor(D)).float()

    # final item embedding，那么核心部分就是model部分
    item_emb, session_emb = model(items, edge_matrix, mask, reversed_sess_item, sess_item, D, A, sess_len)
    # 取出每个会话包含的item对应的embedding，从而得到seq_hidden,它的维度应该是：会话数 * 会话包含item数 * embdding维度
    get = lambda index: item_emb[index][alias_inputs[index]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])

    fusion_sess_emb, scores = model.compute_scores(seq_hidden, mask)
    con_loss = model.SSL(fusion_sess_emb, session_emb)
    # 通过model.compute_scores进行聚合生成会话的表示并预测点击概率
    return targets, scores, 0.005*con_loss


def get_overlap(sessions):
    matrix = np.zeros((len(sessions), len(sessions))) # 每次取100个session, 所以初始化为100 * 100
    for i in range(len(sessions)):
        seq_a = set(sessions[i]) # 为每个session创建集合
        seq_a.discard(0) # 去掉0，保留物品

        # 分配线图中的权重，线图中任意两点（即两个session）之间的边分配权重 W = 两个会话的交集数量 / 两个会话的并集数量
        for j in range(i+1, len(sessions)):
            seq_b = set(sessions[j])
            seq_b.discard(0)
            overlap = seq_a.intersection(seq_b) # 返回一个新集合，其中包含所有集合相同的元素
            ab_set = seq_a | seq_b # 两集合并集
            matrix[i][j] = float(len(overlap))/float(len(ab_set))
            matrix[j][i] = matrix[i][j]

    matrix = matrix + np.diag([1.0]*len(sessions)) # 添加自循环（单位阵）的线图权重矩阵
    degree = np.sum(np.array(matrix), 1) # 线图权重矩阵按行求和即为度
    degree = np.diag(1.0/degree)
    return matrix, degree


def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0

    # 读取数据，对训练数据进行shuffle
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size, drop_last = True,
                                               shuffle=True, pin_memory=True)
    for data in tqdm(train_loader):
        model.optimizer.zero_grad()

        # 重点在于forward函数
        targets, scores, con_loss = forward(model, data)
        targets = trans_to_cuda(targets).long()

        # 计算loss
        loss = model.loss_function(scores, targets - 1)
        loss += con_loss
        loss.backward()
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()

    # 进行测试，指标计算
    print('start predicting: ', datetime.datetime.now())
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=model.batch_size, drop_last = True,
                                              shuffle=False, pin_memory=True)
    result = []
    hit, mrr = [], []
    for data in test_loader:
        targets, scores, con_loss = forward(model, data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        targets = targets.numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))

    result.append(np.mean(hit) * 100)
    result.append(np.mean(mrr) * 100)

    return result

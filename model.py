import datetime
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

class Intra_sess_view(nn.Module):
    def __init__(self, dim, alpha, dropout=0., name=None):
        super(Intra_sess_view, self).__init__()
        self.dim = dim
        self.dropout = dropout
        self.a_0 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_1 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_3 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.bias = nn.Parameter(torch.Tensor(self.dim))

        self.leakyrelu = nn.LeakyReLU(alpha)


    def forward(self, hidden, adj, mask_item=None):
        h = hidden
        batch_size = h.shape[0]
        N = h.shape[1]

        a_input = (h.repeat(1, 1, N).view(batch_size, N * N, self.dim) * h.repeat(1, N, 1)).view(batch_size, N, N, self.dim)

        e_0 = torch.matmul(a_input, self.a_0)
        e_1 = torch.matmul(a_input, self.a_1)
        e_2 = torch.matmul(a_input, self.a_2)
        e_3 = torch.matmul(a_input, self.a_3)

        e_0 = self.leakyrelu(e_0).squeeze(-1).view(batch_size, N, N)
        e_1 = self.leakyrelu(e_1).squeeze(-1).view(batch_size, N, N)
        e_2 = self.leakyrelu(e_2).squeeze(-1).view(batch_size, N, N)
        e_3 = self.leakyrelu(e_3).squeeze(-1).view(batch_size, N, N)

        mask = -9e15 * torch.ones_like(e_0)
        alpha = torch.where(adj.eq(1), e_0, mask)
        alpha = torch.where(adj.eq(2), e_1, alpha)
        alpha = torch.where(adj.eq(3), e_2, alpha)
        alpha = torch.where(adj.eq(4), e_3, alpha)
        alpha = torch.softmax(alpha, dim=-1)

        output = torch.matmul(alpha, h)
        return output


class Inter_sess_view(Module):
    def __init__(self, layers, dataset, emb_size=100):
        super(Inter_sess_view, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
        self.dataset = dataset

    def forward(self, adjacency, embedding):
        item_embeddings = embedding
        item_embedding_layer0 = item_embeddings
        final = [item_embedding_layer0]
        for i in range(self.layers):
            item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), item_embeddings)
            final.append(item_embeddings)
     #  final1 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in final]))
     #  item_embeddings = torch.sum(final1, 0)
        item_embeddings = np.sum(final, 0) / (self.layers+1)
        return item_embeddings


class Sess2Sess_view(Module):
    def __init__(self, layers, batch_size, emb_size=100):
        super(Sess2Sess_view, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.layers = layers

    def forward(self, item_embedding, D, A, session_item, session_len):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        # zeros = torch.zeros([1,self.emb_size])
        item_embedding = torch.cat([zeros, item_embedding], 0)
        seq_h = []
        for i in torch.arange(len(session_item)):
            seq_h.append(torch.index_select(item_embedding, 0, session_item[i]))
        seq_h1 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in seq_h]))
        session_emb_lgcn = torch.div(torch.sum(seq_h1, 1), session_len)
        session = [session_emb_lgcn]
        DA = torch.mm(D, A).float()
        for i in range(self.layers):
            session_emb_lgcn = torch.mm(DA, session_emb_lgcn)
            session.append(session_emb_lgcn)
        session1 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in session]))
        session_emb_lgcn = torch.sum(session1, 0) / (self.layers+1)
        #session_emb_lgcn = np.sum(session, 0) / (self.layers+1)
        return session_emb_lgcn # Θl


# 模型实现核心代码
class CombineGraph(Module):
    def __init__(self, opt, adjacency, num_node, dataset):
        super(CombineGraph, self).__init__()
        self.opt = opt

        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))

        self.adjacency = adjacency
        self.batch_size = opt.batch_size
        self.num_node = num_node
        self.dim = opt.hiddenSize
        self.dropout_local = opt.dropout_local
        self.dropout_global = opt.dropout_global
        self.dropout = nn.Dropout()
        self.layers = opt.layers
        self.dataset = dataset

        self.Inter_sess_view = Inter_sess_view(self.layers, dataset)
        self.Intra_sess_view = Intra_sess_view(self.dim, self.opt.alpha, dropout=0.0)
        self.Sess2Sess = Sess2Sess_view(self.layers, self.batch_size)

        # Item representation & Position representation 初始化
        self.embedding = nn.Embedding(num_node, self.dim)
        self.pos_embedding = nn.Embedding(200, self.dim)

        # Parameters
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))

        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)
        self.linear_transform = nn.Linear(self.dim, self.dim, bias=False)

        self.leakyrelu = nn.LeakyReLU(opt.alpha)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def compute_scores(self, hidden, mask):
        mask = mask.float().unsqueeze(-1)

        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)

        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)

        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask

        fusion_sess_emb = torch.sum(beta * hidden, 1)

        b = self.embedding.weight[1:]
        b = self.dropout(b)
        b = F.normalize(b, dim=-1)
        scores = torch.matmul(fusion_sess_emb, b.transpose(1, 0))

        return fusion_sess_emb, scores


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


    def forward(self, inputs, edgetype, mask, sess_item, reversed_sess_item, sess_len, A, D):
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]
        h = self.embedding(inputs)

        item_emb_Intra = self.Intra_sess_view(h, edgetype, mask)
        item_emb_Inter = self.Inter_sess_view(self.adjacency, self.embedding.weight)
        sess_emb = self.Sess2Sess(self.embedding.weight, D, A, sess_item, sess_len)

        zeros = torch.cuda.FloatTensor(1, self.dim).fill_(0)
        # zeros = torch.zeros(1, self.emb_size)
        item_emb_Inter = torch.cat([zeros, item_emb_Inter], 0)
        get = lambda i: item_emb_Inter[sess_item[i]]
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(sess_item.shape)[1], self.dim).fill_(0)
        # seq_h = torch.zeros(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size)
        for i in torch.arange(sess_item.shape[0]):
            seq_h[i] = get(i)

        item_emb_Intra = F.dropout(item_emb_Intra, self.dropout_local, training=self.training)
        seq_h = F.dropout(seq_h, self.dropout_global, training=self.training)
        final_item_emb =  seq_h + item_emb_Intra

        return sess_emb, final_item_emb


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


def forward(model, data):
    sess_item, alias_inputs, edgetype, items, mask, targets, reversed_sess_item, sess_len = data
    A, D = get_overlap(sess_item.tolist())

    sess_item = trans_to_cuda(sess_item).long()
    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()
    edgetype = trans_to_cuda(edgetype).float()
    mask = trans_to_cuda(mask).long()
    reversed_sess_item = trans_to_cuda(reversed_sess_item).long()
    sess_len = trans_to_cuda(sess_len).long()
    A = trans_to_cuda(torch.Tensor(A)).float()
    D = trans_to_cuda(torch.Tensor(D)).float()

    sess_emb, final_item_emb = model(items, edgetype, mask, sess_item, reversed_sess_item, sess_len, A, D)

    get = lambda index: final_item_emb[index][alias_inputs[index]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])

    fusion_sess_emb, scores = model.compute_scores(seq_hidden, mask)
    con_loss = model.SSL(sess_emb, fusion_sess_emb)

    return targets, scores, 0.01 * con_loss


def get_overlap(sessions):
    matrix = np.zeros((len(sessions), len(sessions)))
    for i in range(len(sessions)):
        seq_a = set(sessions[i])
        seq_a.discard(0)

        for j in range(i+1, len(sessions)):
            seq_b = set(sessions[j])
            seq_b.discard(0)
            overlap = seq_a.intersection(seq_b)
            ab_set = seq_a | seq_b
            matrix[i][j] = float(len(overlap))/float(len(ab_set))
            matrix[j][i] = matrix[i][j]

    matrix = matrix + np.diag([1.0]*len(sessions))
    degree = np.sum(np.array(matrix), 1)
    degree = np.diag(1.0/degree)
    return matrix, degree


def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0

    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size, drop_last = True,
                                               shuffle=True, pin_memory=True)
    for data in tqdm(train_loader):
        model.optimizer.zero_grad()

        targets, scores, con_loss = forward(model, data)
        targets = trans_to_cuda(targets).long()

        # 计算loss
        loss = model.loss_function(scores, targets - 1)
        loss = loss + con_loss
        loss.backward()
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()

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

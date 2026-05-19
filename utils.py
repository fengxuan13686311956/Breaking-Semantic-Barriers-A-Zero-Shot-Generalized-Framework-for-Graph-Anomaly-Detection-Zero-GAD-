import random

import numpy as np
import networkx as nx
import scipy.sparse as sp
import pickle as pkl
import scipy.io as sio
import umap
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.decomposition import PCA

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

import dgl
import torch
import torch.nn as nn
from torch import optim as optim

from dot_gat import DotGAT
from gat import GAT
from gcn import GCN, create_norm
from gin import GIN

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = sparse_mx.shape
    return torch.sparse_coo_tensor(indices, values, shape)

def normalize_score(ano_score):
    ano_score = ((ano_score - np.min(ano_score)) / (np.max(ano_score) - np.min(ano_score)))
    return ano_score

def x_svd(data, out_dim):
    assert data.shape[-1] >= out_dim
    U, S, _ = torch.linalg.svd(data)
    newdata= torch.mm(U[:, :out_dim], torch.diag(S[:out_dim]))
    return newdata

def load_mat(dataset):

    data = sio.loadmat("./Datasets/{}.mat".format(dataset))
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']
    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)
    ano_labels = np.squeeze(np.array(label))

    if 'str_anomaly_label' in data:
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    else:
        str_ano_labels = None
        attr_ano_labels = None
    return adj, feat, ano_labels, str_ano_labels, attr_ano_labels

def random_partition_graph_with_data(adj_matrix, features, num_subgraphs):
    """上面定义的函数：随机划分图，返回子图邻接矩阵和特征列表 + 节点索引"""
    n = adj_matrix.shape[0]
    indices = list(range(n))
    random.shuffle(indices)

    subgraph_indices = [[] for _ in range(num_subgraphs)]
    for i, idx in enumerate(indices):
        subgraph_indices[i % num_subgraphs].append(idx)

    sub_adjs, sub_features = [], []
    for nodes in subgraph_indices:
        nodes = torch.tensor(nodes, dtype=torch.long)
        sub_adj = adj_matrix[nodes][:, nodes]
        sub_feat = features[nodes]
        sub_adjs.append(sub_adj)
        sub_features.append(sub_feat)

    return sub_adjs, sub_features, subgraph_indices

def transformed_features(dataset,adj_dense, features, laplacian_type='standard', normalize_method='standard'):
    """
    合并拉普拉斯矩阵计算与频域特征变换
    Args:
        adj_dense: 密集矩阵格式的邻接矩阵
        features: 原始特征矩阵
        laplacian_type: 拉普拉斯类型 ('standard' 或 'random_walk')
        split_ratio: 频带分割比例
        normalize_method: 归一化方法 ('standard' 或 'minmax')
        scale_low: 低频部分缩放因子
        scale_high: 高频部分缩放因子
    Returns:
        transformed_features: 转换后的特征矩阵
    """

    # 计算拉普拉斯矩阵
    if laplacian_type == 'standard':
        degree = adj_dense.sum(axis=1)
        D = torch.diag(degree)
        L = D - adj_dense
    elif laplacian_type == 'random_walk':
        out_degree = adj_dense.sum(axis=1)
        D_out = torch.diag(out_degree + 1e-8)  # 防止除零
        L = torch.eye(adj_dense.size(0)) - torch.mm(torch.inverse(D_out), adj_dense)
    else:
        raise ValueError(f"Unsupported Laplacian type: {laplacian_type}")

    # 特征分解
    # L = 0.5 * (L + L.T)
    # eps = 1e-5
    # L += eps * torch.eye(L.size(0), device=L.device)
    eigenvalues, eigenvectors = torch.linalg.eigh(L, UPLO='U')
    sorted_idx = torch.argsort(eigenvalues)
    U = eigenvectors[:, sorted_idx]

    # 频域转换
    F = torch.matmul(U.t(), features)

    # 频带归一化
    if normalize_method == 'standard':
        F = (F - F.mean(dim=0)) / (F.std(dim=0) + 1e-8)
    elif normalize_method == 'minmax':
        F = (F - F.min()) / (F.max() - F.min() + 1e-8)

    # 重构特征
    # F = torch.cat([F_low, F_high], dim=0)
    transformed_features = torch.matmul(U, F)
    # print_features(transformed_features)
    return transformed_features

def process_self_loops(adj_mat, mode='none', value=1):
    """
    处理自环操作
    Args:
        adj_mat (scipy.sparse.csr_matrix or torch.Tensor): 输入的邻接矩阵
        mode (str): 自环处理模式，可选值为 'none'（不改动）、'remove'（去自环）、'add'（加自环）
        value (float): 加自环时的值，默认为 1
    Returns:
        adj_mat_processed (scipy.sparse.csr_matrix or torch.Tensor): 处理后的邻接矩阵
    """
    # 如果输入是 PyTorch 张量，将其转换为 SciPy 稀疏矩阵
    if isinstance(adj_mat, torch.Tensor):
        adj_sparse = csr_matrix(adj_mat.cpu().numpy())
    else:
        adj_sparse = adj_mat.copy()  # 确保不修改原矩阵

    if mode == 'remove':
        # 去除自环
        adj_sparse.setdiag(0)
    elif mode == 'add':
        # 添加自环
        adj_sparse.setdiag(1)
    elif mode == 'none':
        # 不做操作
        pass
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # 将处理后的矩阵转换为与输入相同的类型
    if isinstance(adj_mat, torch.Tensor):
        return torch.FloatTensor(adj_sparse.todense()).to(adj_mat.device)
    else:
        return adj_sparse.copy()

def normalize_features(features, method='standard'):
    """
    归一化特征矩阵
    Args:
        features (torch.Tensor): 输入特征矩阵
        method (str): 归一化方法，可选值为 'standard'（标准归一化）或 'minmax'（归一化到 [0, 1] 范围）
    Returns:
        normalized_features (torch.Tensor): 归一化后的特征矩阵
    """
    if method == 'standard':
        # 标准归一化（均值为 0，方差为 1）
        mean = torch.mean(features, dim=0, keepdim=True)
        std = torch.std(features, dim=0, keepdim=True)
        normalized_features = (features - mean) / (std + 1e-8)  # 防止除零错误
    elif method == 'minmax':
        # Min-Max 归一化（缩放到 [0, 1] 范围）
        min_val = torch.min(features, dim=0, keepdim=True).values
        max_val = torch.max(features, dim=0, keepdim=True).values
        normalized_features = (features - min_val) / (max_val - min_val + 1e-8)  # 防止除零错误
    else:
        normalized_features=features

    return normalized_features

def recombine_features(subgraph_indices, processed_features, n):

    d_out = processed_features[0].shape[1]
    device = processed_features[0].device

    new_features = torch.zeros((n, d_out), device=device)

    for nodes, feats in zip(subgraph_indices, processed_features):
        node_tensor = torch.tensor(nodes, dtype=torch.long, device=device)
        new_features[node_tensor] = feats

    return new_features


def loaddata(dataset, args, device):

    # 原始数据加载

    adj, features, ano_label, str_ano_label, attr_ano_label = load_mat(dataset)

    # 邻接矩阵处理
    adj = adj.astype(np.float32)

    adj_dense = torch.FloatTensor(adj.todense())
    # 特征处理
    features = torch.FloatTensor(features.todense())
    features = x_svd(features, args.unifeat)
    # print(features.shape)
    # print(adj_dense.shape)
    features = normalize_features(features, method='none')
    if dataset in ['tfinance','elliptic','questions']:
        n=features.shape[0]
        adj_list,features_list,indices_list=random_partition_graph_with_data(adj_dense,features,5)
        new_features = []
        for features,adj_batch in zip(features_list,adj_list):

            features = transformed_features(dataset, adj_batch, features, laplacian_type='standard',
                                            normalize_method='standard')
            new_features.append(features)
        features=recombine_features(indices_list, new_features, n)
    else:
        features = transformed_features(dataset,adj_dense, features, laplacian_type='standard', normalize_method='standard')

    # 数据类型设备处理
    adj = process_self_loops(adj, mode='none', value=1)
    adj_lil = lil_matrix(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj_lil.tocsr()).to_dense()
    features = features.to(device)
    adj = adj.to(device)
    return adj, features, ano_label


def completionsim(feature1, feature2):
    feature1 = feature1 / torch.norm(feature1, dim=-1, keepdim=True)
    feature2 = feature2 / torch.norm(feature2, dim=-1, keepdim=True)
    dist = torch.sum(feature1*feature2, dim=1)
    dist = dist.detach().cpu().numpy()
    return dist

def evaluate(message, ano_label, str_ano_label=None, attr_ano_label=None):
    score = 1-normalize_score(message)
    auc = roc_auc_score(ano_label, score)
    AP = average_precision_score(ano_label, score, average='macro', pos_label=1, sample_weight=None)

    if str_ano_label is not None:
        sa_auc = roc_auc_score(str_ano_label, score)
        sa_AP = average_precision_score(str_ano_label, score, average='macro', pos_label=1, sample_weight=None)
        print('Structural: AUC: {:.4f} AP:{:.4f}'.format(sa_auc, sa_AP))
    if attr_ano_label is not None:
        aa_auc = roc_auc_score(attr_ano_label, score)
        aa_AP = average_precision_score(attr_ano_label, score, average='macro', pos_label=1, sample_weight=None)
        print('Context: AUC:{:.4f} AP:{:.4f}'.format(aa_auc, aa_AP))
    return auc, AP

def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead,
                 nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "dotgat":
        mod = DotGAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gin":
        mod = GIN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gcn":
        mod = GCN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding")
        )
    elif m_type == "mlp":
        # * just for decoder
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden, out_dim)
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError

    return mod

def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()

    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer


from matplotlib.patches import Circle, Ellipse


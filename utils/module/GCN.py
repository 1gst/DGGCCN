import torch
from torch import nn


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, bias=False):
        super(GraphConvolution, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        nn.init.xavier_uniform_(self.weight)  # xavier初始化，就是论文里的glorot初始化
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, inputs, adj):
        # inputs: (N, n_channels), adj: sparse_matrix (N, N)
        support = torch.matmul(self.dropout(inputs), self.weight)
        output =  torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    def __init__(self, n_features, hidden_dim, dropout, n_classes):
        super(GCN, self).__init__()
        # self.gc1 = GraphConvolution(n_features, hidden_dim, dropout)
        # self.gc2 = GraphConvolution(hidden_dim, n_classes, dropout)

        self.gc1 = GraphConvolution(n_features, hidden_dim, dropout)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim, dropout)
        self.gc3 = GraphConvolution(hidden_dim, n_classes, dropout)
        # self.gc4 = GraphConvolution(hidden_dim, hidden_dim, dropout)
        # self.gc5 = GraphConvolution(hidden_dim, n_classes, dropout)
        # self.gc6 = GraphConvolution(hidden_dim, n_classes, dropout)
        self.relu = nn.RReLU()

    def forward(self, inputs, adj):
        x = inputs
        # x = self.relu(self.gc1(x, adj))
        # x = self.gc2(x, adj)

        x = self.relu(self.gc1(x, adj))
        # x = self.relu(self.gc2(x, adj))
        diag = torch.diag_embed(torch.diag(adj))
        nodes = (adj - diag == 1).any(dim=1).nonzero().squeeze().tolist()
        if type(nodes) is int:
            nodes = [nodes]
        new_adj = torch.zeros_like(adj) + diag
        for i in nodes:
            new_adj[:, i] = 1
        x = self.gc2(x, new_adj)
        # x = self.relu(self.gc3(x, adj))
        # x = self.relu(self.gc4(x, adj))
        # x = self.gc5(x, adj)
        # x = self.relu(self.gc5(x, adj))
        # x=self.gc6(x, adj)
        return x
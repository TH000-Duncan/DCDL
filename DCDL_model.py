import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix,  diags
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor

import dgl
from dgl.nn import GATConv
from scipy.sparse import coo_matrix, vstack, hstack
from DCDL import DCDL


def Split_Graph_to_device(G, device, split_num=16):
    G_list = []
    length = G.shape[0] // split_num  
    for i in range(split_num):
        if i == split_num - 1:
            G_list.append(G[length * i : G.shape[0]]) 
        else:
           G_list.append(G[length * i : length * (i + 1)])
    G_split = [SparseTensor.from_scipy(G_i).to(device) for G_i in G_list] 
    return G_split


def normalize_G(G):

    def normalize_f(matrix):
        norm = np.sqrt((matrix.data**2).sum())
        if (norm == 0):
            return matrix
        return  matrix / norm

    norm_G = normalize_f(G@G.T)

    return norm_G



def mix_r_graph(raw_graph, threshold=4):
    ui_graph, bi_graph, ub_graph = raw_graph
    uu_graph = ub_graph @ ub_graph.T
    uu_graph.data = np.where(uu_graph.data > threshold, 1, 0)

    bb_graph = ub_graph.T @ ub_graph
    bb_graph.data = np.where(bb_graph.data > threshold, 1, 0)

    r_graph = sp.vstack([ub_graph, bb_graph])
    r_graph = sp.hstack([r_graph, sp.vstack([uu_graph, ub_graph.T])])
    
    r_graph = r_graph.tocsr()

    return r_graph

def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values),
                                     torch.Size(graph.shape))
    return graph

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

class DCDL(nn.Module):
    def __init__(self, raw_graph, device, dp, l2_norm, emb_size=64):
        super().__init__()


        self.ui_graph, self.bi_graph, self.ub_graph = raw_graph
        self.device = device
        self.num_users, self.num_bundles, self.num_items = (
            self.ub_graph.shape[0],
            self.ub_graph.shape[1],
            self.ui_graph.shape[1],
        )
        G = mix_r_graph(raw_graph)
        self.r_graph = Split_Graph_to_device(normalize_G(G), device)

        # embeddings
        self.users_feature = nn.Parameter(
            torch.FloatTensor(self.num_users, emb_size).normal_(0, 0.5 / emb_size)
        )
        self.bundles_feature = nn.Parameter(
            torch.FloatTensor(self.num_bundles, emb_size).normal_(0, 0.5 / emb_size)
        )
        self.items_feature = nn.Parameter(
            torch.FloatTensor(self.num_items, emb_size).normal_(0, 0.5 / emb_size)
        )
        

        num_users, num_bundles = self.ub_graph.shape

        zero_matrix_1 = coo_matrix((num_users, num_users), dtype=int)
        zero_matrix_2 = coo_matrix((num_bundles, num_bundles), dtype=int)

        upper_matrix = hstack([self.ub_graph, zero_matrix_1])
        lower_matrix = hstack([zero_matrix_2, self.ub_graph.T])
        new_ub_graph = vstack([upper_matrix, lower_matrix])

        dgl_ub_graph = dgl.from_scipy(new_ub_graph)
        dgl_ub_graph = dgl.add_self_loop(dgl_ub_graph)
        self.dgl_ub_graph = dgl_ub_graph.to(device)

        self.gat_layer1 = GATConv(in_feats=emb_size, out_feats=emb_size, num_heads=1, feat_drop=0.2, attn_drop=0.2)


        i_u_i = self.ui_graph.T @ self.ui_graph
        i_b_i = self.bi_graph.T @ self.bi_graph

        i_u_i.data[i_u_i.data <= 1] = 0
        i_b_i.data[i_b_i.data <= 1] = 0

        A_i = sp.diags(1 / ((i_u_i.sum(axis=1) + 1e-8).A.ravel())) @ i_u_i
        B_i = sp.diags(1 / ((i_b_i.sum(axis=1) + 1e-8).A.ravel())) @ i_b_i

        self.A_i = to_tensor(A_i).to(self.device)
        self.B_i = to_tensor(B_i).to(self.device)

        ui_norm = sp.diags(1 / ((self.ui_graph.sum(axis=1) + 1e-8).A.ravel())) @ self.ui_graph
        bi_norm = sp.diags(1 / ((self.bi_graph.sum(axis=1) + 1e-8).A.ravel())) @ self.bi_graph

        self.bi_avg = to_tensor(bi_norm).to(self.device)
        self.ui_avg = to_tensor(ui_norm).to(self.device)


        self.timesteps = 15
        self.denoise_model = DCDL.CFD_restore(emb_size, self.num_items, 30, 0.1, 'mlp1', device)
        self.diffusion = DCDL.CFD_diffusion(self.timesteps, 0.0001, 0.02, 2.0)
        
        self.user_bound = nn.Parameter(
            torch.FloatTensor(emb_size, 1).normal_(0, 0.5 / emb_size)
        )

        self.drop = nn.Dropout(dp)
        self.embed_L2_norm = l2_norm
    

    def Intention(self):
        items_iui = F.relu(torch.matmul(self.A_i, self.items_feature))
        items_ibi = F.relu(torch.matmul(self.B_i, self.items_feature))
        items_feature = items_iui + items_ibi + self.items_feature
        return items_feature

    def propagate(self):

        embed_0 = torch.cat([self.users_feature, self.bundles_feature], dim=0)

        gat_output1 = self.gat_layer1(self.dgl_ub_graph, embed_0)
        gat_output1_mean = torch.mean(gat_output1, dim=1)

        embed_0 = gat_output1_mean
        embed_r = torch.cat([G @ embed_0 for G in self.r_graph], dim=0)

        ub_embeds = self.drop(embed_r)
        users_on_ub, bundles_on_ub = torch.split(
            ub_embeds, [self.num_users, self.num_bundles], dim=0
        )

        items_feature = self.Intention()
        bundles_on_bi = F.relu(torch.matmul(self.bi_avg, items_feature))
        bundles_feature = bundles_on_bi + bundles_on_ub
        users_on_ui = F.relu(torch.matmul(self.ui_avg, items_feature))
        users_feature = users_on_ui + users_on_ub


        x_start = users_on_ub.squeeze(1)
        h = users_feature
        n = torch.randint(0, self.timesteps, (self.num_users,), device=self.device).long()
        diff_loss, predicted_x = self.diffusion.p_losses(denoise_model=self.denoise_model, x_start=x_start, h=h, t=n,
                                                         loss_type='l2')

        return users_feature, bundles_feature, items_feature, diff_loss, predicted_x


    def predict(self, users_feature, bundles_feature):
        pred = torch.sum(users_feature * bundles_feature, 2)
        return pred

    def regularize(self, users_feature, bundles_feature, items_feature):
        loss = self.embed_L2_norm * (
            (users_feature ** 2).sum() + (bundles_feature ** 2).sum() + (items_feature ** 2).sum()
        )
        return loss

    def forward(self, users, bundles):
        users_feature, bundles_feature, items_feature, diff_loss, predicted_x = self.propagate()
        users_embedding = users_feature[users].expand(-1, bundles.shape[1], -1)
        bundles_embedding = bundles_feature[bundles]
        pred = self.predict(users_embedding, bundles_embedding)
        reg_loss = self.regularize(users_feature, bundles_feature, items_feature)
        user_score_bound = predicted_x[users] @ self.user_bound
        return pred, user_score_bound, reg_loss, diff_loss, self.users_feature

    def evaluate(self, propagate_result, users):
        users_feature, bundles_feature, items_feature, diff_loss, predicted_x = propagate_result
        users_feature = users_feature[users]
        scores = users_feature @ (bundles_feature.T)
        return scores

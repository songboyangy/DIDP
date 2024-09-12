import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn import GCNConv
from Optim import ScheduledOptim

from TransformerBlock import TransformerBlock
from models.ConvBlock import *

'''To GPU'''
def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

'''To CPU'''
def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

'''Mask previous activated users'''
def get_previous_user_mask(seq, user_size):
    assert seq.dim() == 2
    prev_shape = (seq.size(0), seq.size(1), seq.size(1))
    seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
    previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
    previous_mask = torch.from_numpy(previous_mask)
    if seq.is_cuda:
        previous_mask = previous_mask.cuda()
    masked_seq = previous_mask * seqs.data.float()

    # force the 0th dimension (PAD) to be masked
    PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
    if seq.is_cuda:
        PAD_tmp = PAD_tmp.cuda()
    masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
    ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
    if seq.is_cuda:
        ans_tmp = ans_tmp.cuda()
    masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float('-inf'))
    return masked_seq.cuda()






class Fusion(nn.Module):
    def __init__(self, input_size, out=1, dropout=0.2):
        super(Fusion, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, out)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def forward(self, hidden, dy_emb):
        emb = torch.cat([hidden.unsqueeze(dim=0), dy_emb.unsqueeze(dim=0)], dim=0)
        emb_score = F.softmax(self.linear2(torch.tanh(self.linear1(emb))), dim=0)
        emb_score = self.dropout(emb_score)
        out = torch.sum(emb_score * emb, dim=0)
        return out

class GraphNN(nn.Module):
    def __init__(self, ntoken, ninp, dropout=0.5, is_norm=True):
        super(GraphNN, self).__init__()
        self.embedding = nn.Embedding(ntoken, ninp, padding_idx=0)
        # in:inp,out:nip*2
        self.gnn1 = GCNConv(ninp, ninp * 2)
        self.gnn2 = GCNConv(ninp * 2, ninp)
        self.is_norm = is_norm

        self.dropout = nn.Dropout(dropout)
        if self.is_norm:
            self.batch_norm = torch.nn.BatchNorm1d(ninp)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.embedding.weight)

    def forward(self, graph):
        graph_edge_index = graph.edge_index.cuda()
        graph_x_embeddings = self.gnn1(self.embedding.weight, graph_edge_index)
        graph_x_embeddings = self.dropout(graph_x_embeddings)
        graph_output = self.gnn2(graph_x_embeddings, graph_edge_index)
        if self.is_norm:
            graph_output = self.batch_norm(graph_output)
        # print(graph_output.shape)
        return graph_output.cuda()

class LSTMGNN(nn.Module):
    def __init__(self, hypergraphs, args, dropout=0.2):
        super(LSTMGNN, self).__init__()

        # parameters
        self.emb_size = args.embSize
        self.n_node = args.n_node
        self.layers = args.layer
        self.dropout = nn.Dropout(dropout)
        self.drop_rate = dropout
        self.n_channel = len(hypergraphs) 
        self.win_size = 5

        #hypergraph
        self.H_Item = hypergraphs[0]   
        self.H_User =hypergraphs[1]

        self.gnn = GraphNN(self.n_node, self.initial_feature, dropout=dropout)
        self.fus = Fusion(self.emb_size)
        self.decoder_attention=TransformerBlock(input_size=self.emb_size, n_heads=8)


        ###### user embedding
        self.user_embedding = nn.Embedding(self.n_node, self.emb_size, padding_idx=0)


        self.linear = nn.Linear(self.emb_size*3, self.emb_size)
        self.linear1=nn.Linear(self.emb_size, self.n_node)

        self.reset_parameters()

        # #### optimizer and loss function
        # self.optimizerAdam = torch.optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-09)
        # self.optimizer = ScheduledOptim(self.optimizerAdam, self.emb_size, args.n_warmup_steps)
        # self.loss_function = nn.CrossEntropyLoss(size_average=False, ignore_index=0)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def self_gating(self, em, channel):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.weights[channel]) + self.bias[channel]))






    def _dropout_graph(self, graph, keep_prob):
        size = graph.size()
        index = graph.coalesce().indices().t()
        values = graph.coalesce().values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    '''social structure and hypergraph structure embeddding'''
    def structure_embed(self, H_Time=None, H_Item=None, H_User=None):

        if self.training:
            H_Item = self._dropout_graph(self.H_Item, keep_prob=1-self.drop_rate)
            H_User = self._dropout_graph(self.H_User, keep_prob=1-self.drop_rate)
        else:
            H_Item = self.H_Item
            H_User = self.H_User

        u_emb_c2 = self.self_gating(self.user_embedding.weight, 0)


        all_emb_c2 = [u_emb_c2]


        for k in range(self.layers):
            # Channel Item
            u_emb_c2 = torch.sparse.mm(H_Item, u_emb_c2)
            norm_embeddings2 = F.normalize(u_emb_c2, p=2, dim=1)
            all_emb_c2 += [norm_embeddings2]


        u_emb_c2 = torch.stack(all_emb_c2, dim=1)
        u_emb_c2 = torch.sum(u_emb_c2, dim=1)

        return u_emb_c2

    def forward(self, input, label,socail_graph,diff_model,social_reverse_model,cas_reverse_model):

        mask = (input == 0)
        mask_label = (label == 0)

        '''structure embeddding'''
        HG_Uemb = self.structure_embed()

        '''past cascade embeddding'''
        cas_seq_emb = F.embedding(input, HG_Uemb)

        social_embedding=self.dropout(self.gnn(socail_graph))

        social_seq_emb= F.embedding(input,social_embedding)

        noise_cas_emb, noise_social_emb, ts, pt = self.apply_noise(cas_seq_emb, social_seq_emb,diff_model)  # 向embedding中加入了噪声
        social_model_output = social_reverse_model(noise_social_emb, ts)  # 在后向的过程中添加了监督信号，来辅助他的重构，因为要构建两个所以也不方便来做回传
        cas_model_output = cas_reverse_model(noise_cas_emb,ts)

        social_recons = diff_model.get_reconstruct_loss(social_seq_emb, social_model_output, pt)
        cas_recons = diff_model.get_reconstruct_loss(cas_seq_emb, cas_model_output, pt)
        recons_loss = (social_recons + cas_recons) / 2
        user_seq_emb=self.fus(social_model_output,cas_model_output)
        att_out=self.decoder_attention(user_seq_emb,user_seq_emb,user_seq_emb,mask=mask)
        prediction=self.linear1(att_out)

        mask = get_previous_user_mask(input, self.n_node)

        return (prediction + mask).view(-1, prediction.size(-1)).cuda(),recons_loss

    def model_prediction(self, input):

        mask = (input == 0)

        '''structure embeddding'''
        HG_Uemb = self.structure_embed()

        '''cascade embeddding'''
        cas_seq_emb = F.embedding(input, HG_Uemb)

        ####long-term temporal influence
        source_emb = cas_seq_emb[:, 0, :]
        L_cas_emb = self.long_term_att(source_emb, cas_seq_emb, cas_seq_emb, mask=mask.cuda())

        ####short-term temporal influence
        user_cas_gru, _ = self.past_gru(cas_seq_emb)
        user_cas_lstm, _ = self.past_lstm(cas_seq_emb)
        S_cas_emb = self.short_term_att(user_cas_gru, user_cas_lstm, user_cas_lstm, mask=mask.cuda())

        LS_cas_emb = torch.cat([cas_seq_emb, L_cas_emb, S_cas_emb], -1)
        LS_cas_emb = self.linear(LS_cas_emb)

        output = self.past_multi_att(LS_cas_emb, LS_cas_emb, LS_cas_emb, mask)
        pre_y = torch.matmul(output, torch.transpose(HG_Uemb, 1, 0))
        mask = get_previous_user_mask(input, self.n_node)

        return (pre_y + mask).view(-1, pre_y.size(-1)).cuda()

    def model_prediction2(self, input, input_len, HG_Time, HG_Item, HG_User):

        mask = (input == 0)

        b = input_len.size(0)

        '''structure embeddding'''
        HG_Uemb = self.structure_embed(H_Time=HG_Time, H_Item=HG_Item, H_User=HG_User)

        '''cascade embeddding'''
        cas_seq_emb = F.embedding(input, HG_Uemb)

        ####long-term temporal influence
        source_emb = cas_seq_emb[:, 0, :]
        L_cas_emb = self.long_term_att(source_emb, cas_seq_emb, cas_seq_emb, mask=mask.cuda())

        ####short-term temporal influence
        user_cas_gru, _ = self.past_gru(cas_seq_emb)
        user_cas_lstm, _ = self.past_lstm(cas_seq_emb)
        S_cas_emb = self.short_term_att(user_cas_gru, user_cas_lstm, user_cas_lstm, mask=mask.cuda())

        LS_cas_emb = torch.cat([cas_seq_emb, L_cas_emb, S_cas_emb], -1)
        LS_cas_emb = self.linear(LS_cas_emb)

        output = self.past_multi_att(LS_cas_emb, LS_cas_emb, LS_cas_emb, mask)
        pre_y = torch.matmul(output, torch.transpose(HG_Uemb, 1, 0))
        mask = get_previous_user_mask(input, self.n_node)

        return (pre_y + mask).view(-1, pre_y.size(-1)).cuda()

    def apply_noise(self, user_emb, item_emb, diff_model):
        # cat_emb shape: (batch_size*3, emb_size)
        emb_size = user_emb.shape[0]
        ts, pt = diff_model.sample_timesteps(emb_size, 'uniform')
        # ts_ = torch.tensor([self.config['steps'] - 1] * cat_emb.shape[0]).to(cat_emb.device)

        # add noise to users
        user_noise = torch.randn_like(user_emb)
        item_noise = torch.randn_like(item_emb)
        user_noise_emb = diff_model.q_sample(user_emb, ts, user_noise)
        item_noise_emb = diff_model.q_sample(item_emb, ts, item_noise)
        return user_noise_emb, item_noise_emb, ts, pt




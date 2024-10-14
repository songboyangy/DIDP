import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn import GCNConv
from models.TransformerBlock import TransformerBlock


def trans_to_cuda(variable, device_id=0):
    if torch.cuda.is_available():
        return variable.cuda(device_id)
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def get_previous_user_mask(seq, user_size):
    assert seq.dim() == 2
    device = seq.device

    prev_shape = (seq.size(0), seq.size(1), seq.size(1))
    seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))


    previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
    previous_mask = torch.from_numpy(previous_mask).to(device)
    masked_seq = previous_mask * seqs.data.float()
    PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1).to(device)
    masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
    ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size).to(device)
    masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float('-inf'))

    return masked_seq.to(device)





class GraphNN(nn.Module):
    def __init__(self, ntoken, ninp,device, dropout=0.5, is_norm=True):
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
        self.device=device

    def init_weights(self):
        init.xavier_normal_(self.embedding.weight)

    def forward(self, graph):
        graph_edge_index = graph.edge_index.to(self.device)
        graph_x_embeddings = self.gnn1(self.embedding.weight, graph_edge_index)
        graph_x_embeddings = self.dropout(graph_x_embeddings)
        graph_output = self.gnn2(graph_x_embeddings, graph_edge_index)
        if self.is_norm:
            graph_output = self.batch_norm(graph_output)

        return graph_output.to(self.device)

class DIDP(nn.Module):
    def __init__(self, hypergraphs, args, dropout=0.2):
        super(DIDP, self).__init__()

        self.args=args

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

        self.gnn = GraphNN(self.n_node, self.emb_size, dropout=dropout,device=args.device)


        self.dropout = nn.Dropout(p=0.1)
        self.social_attention=TransformerBlock(input_size=self.emb_size, n_heads=8,device=args.device)
        self.cas_attention=TransformerBlock(input_size=self.emb_size, n_heads=8,device=args.device)
        self.linear_social=nn.Linear(self.emb_size, self.n_node)
        self.linear_diff=nn.Linear(self.emb_size, self.n_node)

        ###### user embedding
        self.user_embedding = nn.Embedding(self.n_node, self.emb_size, padding_idx=0)


        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)






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


        u_emb_c2=self.user_embedding.weight


        all_emb_c2 = [u_emb_c2]


        for k in range(self.layers):
            u_emb_c2 = torch.sparse.mm(H_Item, u_emb_c2)
            norm_embeddings2 = F.normalize(u_emb_c2, p=2, dim=1)
            all_emb_c2 += [norm_embeddings2]


        u_emb_c2 = torch.stack(all_emb_c2, dim=1)
        u_emb_c2 = torch.sum(u_emb_c2, dim=1)

        return u_emb_c2

    def forward(self, input,cas_time,label, social_graph,diff_model,cas_reverse_model,train=True):

        mask = (input == 0)



        '''structure embeddding'''
        HG_Uemb = self.structure_embed()

        '''past cascade embeddding'''
        cas_seq_emb = F.embedding(input, HG_Uemb)

        social_embedding=self.dropout(self.gnn(social_graph))

        social_seq_emb= F.embedding(input,social_embedding)
        tensor_size=social_seq_emb.size()
        batch_size = tensor_size[0]
        seq_len = tensor_size[1]

        social_seq_emb_reshaped = social_seq_emb.view(-1, tensor_size[-1])
        cas_seq_emb_reshaped = cas_seq_emb.view(-1, tensor_size[-1])


        if train:
            noise_cas_emb,  ts, pt = self.apply_noise2(social_seq_emb_reshaped, diff_model,tensor_size)

            social_model_output = cas_reverse_model(noise_cas_emb,ts)
            social_recons = diff_model.get_reconstruct_loss(social_seq_emb_reshaped, social_model_output, pt)

            recons_loss = torch.mean(social_recons)

        else:
            social_model_output = diff_model.p_sample(cas_reverse_model, social_seq_emb_reshaped, self.args.sampling_steps,
                                                  self.args.sampling_noise)
        ssl=self.calc_ssl_sim(social_model_output,cas_seq_emb_reshaped,self.args.tau)

        social_model_output1=social_model_output.view(batch_size, seq_len, -1)
        social_model_output2 = self.diff_fuse(torch.cat([social_model_output1, social_seq_emb], dim=-1))
        social_cas = self.social_attention(social_model_output2, social_model_output2, social_model_output2)
        social_prediction = torch.matmul(social_cas, torch.transpose(social_embedding, 1, 0))
        output_s1=self.linear_social(social_cas)
        output_s=social_prediction+output_s1

        diff_cas = self.cas_attention(cas_seq_emb, cas_seq_emb,cas_seq_emb)
        diff_prediction=torch.matmul(diff_cas, torch.transpose(HG_Uemb, 1, 0))
        output_d1=self.linear_diff(diff_cas)
        output_d=diff_prediction+output_d1

        prediction=output_s+output_d

        mask = get_previous_user_mask(input, self.n_node)
        result = (prediction + mask).view(-1, prediction.size(-1)).to(self.args.device)
        if train:
            return result, recons_loss, ssl
        else:
            return result

    def calc_ssl_sim(self, emb1, emb2, tau, normalization=False):
        # (emb1, emb2) = (F.normalize(emb1, p=2, dim=0), F.normalize(emb2, p=2, dim=0))\
        if normalization:
            emb1 = nn.functional.normalize(emb1, p=2, dim=1, eps=1e-12)
            emb2 = nn.functional.normalize(emb2, p=2, dim=1, eps=1e-12)

        (emb1_t, emb2_t) = (emb1.t(), emb2.t())  # 这个得到他们的转置矩阵

        pos_scores_users = torch.exp(torch.div(F.cosine_similarity(emb1, emb2, dim=1, eps=1e-8), tau))  # Sum by row
        # denominator cosine_similarity: following codes
        if self.args.inter:

            denominator_scores = torch.mm(emb1, emb2_t)
            norm_emb1 = torch.norm(emb1, dim=-1)
            norm_emb2 = torch.norm(emb2, dim=-1)
            norm_emb = torch.mm(norm_emb1.unsqueeze(1), norm_emb2.unsqueeze(1).t())

            denominator_scores1 = torch.exp(torch.div(denominator_scores / norm_emb, tau)).sum(1)  # Sum by row
            denominator_scores2 = torch.exp(torch.div(denominator_scores / norm_emb, tau)).sum(0)  # Sum by column
            # denominator cosine_similarity: above codes

            ssl_loss1 = -torch.mean(torch.log(pos_scores_users / denominator_scores1))
            ssl_loss2 = -torch.mean(torch.log(pos_scores_users / denominator_scores2))
        else:  # interAintra
            denominator_scores = torch.mm(emb1, emb2_t)
            norm_emb1 = torch.norm(emb1, dim=-1)
            norm_emb2 = torch.norm(emb2, dim=-1)
            norm_emb = torch.mm(norm_emb1.unsqueeze(1), norm_emb2.unsqueeze(1).t())
            denominator_scores1 = torch.exp(torch.div(denominator_scores / norm_emb, tau)).sum(1)  # Sum by row
            denominator_scores2 = torch.exp(torch.div(denominator_scores / norm_emb, tau)).sum(0)  # Sum by column

            denominator_scores_intraview1 = torch.mm(emb1, emb1_t)
            norm_intra1 = torch.mm(norm_emb1.unsqueeze(1), norm_emb1.unsqueeze(1).t())
            denominator_intra_scores1 = torch.exp(torch.div(denominator_scores_intraview1 / norm_intra1, tau))
            diag1 = torch.diag(denominator_intra_scores1)
            d_diag1 = torch.diag_embed(diag1)
            denominator_intra_scores1 = denominator_intra_scores1 - d_diag1  # here we set the elements on diagonal to be 0.
            intra_denominator_scores1 = denominator_intra_scores1.sum(1)  # Sum by row#
            # .sum(1)

            denominator_scores_intraview2 = torch.mm(emb2, emb2_t)
            norm_intra2 = torch.mm(norm_emb2.unsqueeze(1), norm_emb2.unsqueeze(1).t())
            denominator_intra_scores2 = torch.exp(torch.div(denominator_scores_intraview2 / norm_intra2, tau))
            diag2 = torch.diag(denominator_intra_scores2)
            d_diag2 = torch.diag_embed(diag2)
            denominator_intra_scores2 = denominator_intra_scores2 - d_diag2
            intra_denominator_scores2 = denominator_intra_scores2.sum(1)

            # denominator cosine_similarity: above codes
            ssl_loss1 = -torch.mean(torch.log(pos_scores_users / (denominator_scores1 + intra_denominator_scores1)))
            ssl_loss2 = -torch.mean(torch.log(pos_scores_users / (denominator_scores2 + intra_denominator_scores2)))

        return ssl_loss1 + ssl_loss2

    def apply_noise2(self, user_emb, diff_model,seq_size):

        batch_size, seq_len, emb_size = seq_size
        ts, pt = diff_model.sample_timesteps(batch_size, 'uniform')
        ts_expanded = ts.unsqueeze(1).repeat(1, seq_len).view(-1)

        user_noise = torch.randn_like(user_emb)

        user_noise_emb = diff_model.forward_process(user_emb,ts_expanded, user_noise)

        return user_noise_emb, ts_expanded, pt


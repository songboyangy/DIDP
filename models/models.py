import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn import GCNConv
from triton.language import tensor
from utils.utils import PositionalEncoding,LearnedPositionalEncoding,TimeAttention,TimeDifferenceEncoder,cumulative_average,mse_loss
from Optim import ScheduledOptim

from models.TransformerBlock import TransformerBlock
from models.ConvBlock import *

'''To GPU'''
def trans_to_cuda(variable, device_id=0):
    if torch.cuda.is_available():
        return variable.cuda(device_id)
    else:
        return variable


'''To CPU'''
def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

# '''Mask previous activated users'''
# def get_previous_user_mask(seq, user_size):
#     assert seq.dim() == 2
#     prev_shape = (seq.size(0), seq.size(1), seq.size(1))
#     seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
#     previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
#     previous_mask = torch.from_numpy(previous_mask)
#     if seq.is_cuda:
#         previous_mask = previous_mask.cuda()
#     masked_seq = previous_mask * seqs.data.float()
#
#     # force the 0th dimension (PAD) to be masked
#     PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
#     if seq.is_cuda:
#         PAD_tmp = PAD_tmp.cuda()
#     masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
#     ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
#     if seq.is_cuda:
#         ans_tmp = ans_tmp.cuda()
#     masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float('-inf'))
#     return masked_seq.cuda()


def get_previous_user_mask(seq, user_size):
    assert seq.dim() == 2
    device = seq.device  # 获取 seq 的设备

    prev_shape = (seq.size(0), seq.size(1), seq.size(1))
    seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))

    # 使用 numpy 生成 lower-triangular mask，并将其转为 PyTorch 张量
    previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
    previous_mask = torch.from_numpy(previous_mask).to(device)  # 移动到与 seq 相同的设备

    # 应用 mask
    masked_seq = previous_mask * seqs.data.float()

    # 强制将 0 维 (PAD) 设置为掩码
    PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1).to(device)  # 移动到与 seq 相同的设备
    masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)

    # 创建一个全零的张量并将 masked_seq 散布到它上面
    ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size).to(device)  # 移动到与 seq 相同的设备
    masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float('-inf'))

    return masked_seq.to(device)


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


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        # 定义第一层全连接层，输入大小为 input_size，输出大小为 hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 定义第二层全连接层，输入大小为 hidden_size，输出大小为 output_size
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 第一层：线性变换后加上ReLU激活函数
        x = F.relu(self.fc1(x))
        # 第二层：线性变换后直接输出
        x = self.fc2(x)
        return x



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
        # print(graph_output.shape)
        return graph_output.to(self.device)

class LSTMGNN(nn.Module):
    def __init__(self, hypergraphs, args, dropout=0.2):
        super(LSTMGNN, self).__init__()

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
        self.fus = Fusion(self.emb_size)
        self.fus1=Fusion(self.emb_size)
        self.fus2=Fusion(self.emb_size)
        self.decoder_attention=TransformerBlock(input_size=self.emb_size, n_heads=8,device=args.device)
        #self.pos_encoding = PositionalEncoding(self.emb_size)
        self.pos_encoding = LearnedPositionalEncoding(embedding_size=self.emb_size)
        self.time_diff_encoder=TimeDifferenceEncoder(dimension=self.emb_size).to(args.device)
        self.time_user_cat=nn.Linear(self.emb_size*2,self.emb_size)
        self.time_attention=TimeAttention(200,self.emb_size)
        self.dropout = nn.Dropout(p=0.1)

        ### channel self-gating parameters
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.emb_size, self.emb_size)) for _ in range(self.n_channel)])
        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(1, self.emb_size)) for _ in range(self.n_channel)])


        ###### user embedding
        self.user_embedding = nn.Embedding(self.n_node, self.emb_size, padding_idx=0)

        self.temp_lstm=nn.LSTM(self.emb_size, self.emb_size, batch_first=True)
        self.linear = nn.Linear(self.emb_size*2, self.emb_size)
        self.linear1=nn.Linear(self.emb_size, self.n_node)
        self.linear2=nn.Linear(self.emb_size*2, self.emb_size)

        self.social_mlp=MLP(self.emb_size,int(self.emb_size/2),self.emb_size)
        self.cas_mlp=MLP(self.emb_size,int(self.emb_size/2),self.emb_size)

        self.reset_parameters()


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

    def forward(self, input,cas_time,label, social_graph,diff_model,cas_reverse_model,train=True):

        mask = (input == 0)
        #label=input[:,1:]



        '''structure embeddding'''
        HG_Uemb = self.structure_embed()

        '''past cascade embeddding'''
        cas_seq_emb = F.embedding(input, HG_Uemb)

        label_embedding=F.embedding(label, HG_Uemb)

        social_embedding=self.dropout(self.gnn(social_graph))

        social_seq_emb= F.embedding(input,social_embedding)
        tensor_size=social_seq_emb.size()
        batch_size = tensor_size[0]
        seq_len = tensor_size[1]
        #下面的过程确实需要修改，需要将emd展开成二维的，这也是最简单的方法，展开成二维之后，再重塑成三维的
        social_seq_emb_reshaped = social_seq_emb.view(-1, tensor_size[-1])
        cas_seq_emb_reshaped = cas_seq_emb.view(-1, tensor_size[-1])
        #label_embedding=label_embedding.view(-1, tensor_size[-1])

        if train:
            ssl=self.calc_ssl_sim(social_seq_emb_reshaped,cas_seq_emb_reshaped,self.args.tau)
            #ssl=self.social_cas_ssl(social_seq_emb_reshaped,cas_seq_emb_reshaped)

            noise_cas_emb,  ts, pt = self.apply_noise2(cas_seq_emb_reshaped, diff_model,tensor_size)

            cas_model_output = cas_reverse_model(noise_cas_emb,ts)
            cas_recons = diff_model.get_reconstruct_loss(cas_seq_emb_reshaped, cas_model_output, pt)
            #ssl = self.calc_ssl_sim(social_seq_emb_reshaped, cas_model_output, self.args.tau)
            # sim1=(cas_seq_emb_reshaped*label_embedding).sum(dim=1)
            # sim2=(cas_model_output*label_embedding).sum(dim=1)
            # difference = (sim1 - sim2)**2
            # # 对差值求平均
            # mean_difference = difference.mean()



            recons_loss = torch.mean(cas_recons)
            # recons_loss=recons_loss+0.5*mean_difference

        else:
            cas_model_output = diff_model.p_sample(cas_reverse_model, cas_seq_emb_reshaped, self.args.sampling_steps,
                                                  self.args.sampling_noise)

        cas_model_output1=cas_model_output.view(batch_size, seq_len, -1)



        #cas_model_output2 =self.linear(torch.cat([cas_model_output1,cas_seq_emb],dim=-1))
        cas_model_output2=self.fus2(cas_model_output1,cas_seq_emb)
        #cas_model_output2=self.dropout(cas_model_output2)



        user_seq_emb = self.fus(social_seq_emb, cas_model_output2)

        #user_seq_emb = self.linear2(torch.cat([social_seq_emb, cas_model_output2],dim=-1))
        #user_seq_emb = self.fus(social_seq_emb, cas_seq_emb)
        # time_diff_embedding=self.time_diff_emb(cas_time)
        # user_seq_emb=self.time_user_cat(torch.cat([user_seq_emb,time_diff_embedding],dim=-1))
        #user_seq_emb=self.pos_encoding(user_seq_emb)

        #cas_tem,_=self.temp_lstm(user_seq_emb)


        att_out=self.decoder_attention(user_seq_emb,user_seq_emb,user_seq_emb,mask=mask)
        #att_out=self.linear(torch.cat([att_out,cas_model_output2],dim=-1))
        #att_out = F.relu(self.linear(torch.cat([att_out, cas_model_output1], dim=-1)))
        #cas_out= self.fus1(cas_tem,att_out)

        prediction = self.linear1(att_out)
        #prediction = self.linear1(cas_out)

        mask = get_previous_user_mask(input, self.n_node)
        result = (prediction + mask).view(-1, prediction.size(-1)).to(self.args.device)
        if train:
            c1=cumulative_average(cas_model_output1)
            c_difference=mse_loss(y_pred=c1,y_true=label_embedding)
            recons_loss=recons_loss+c_difference
            return result, recons_loss, ssl
        else:
            return result

    def apply_noise(self, user_emb, item_emb, diff_model):
        # cat_emb shape: (batch_size*3, emb_size)
        emb_size = user_emb.shape[0]
        ts, pt = diff_model.sample_timesteps(emb_size, 'uniform')
        # ts_ = torch.tensor([self.config['steps'] - 1] * cat_emb.shape[0]).to(cat_emb.device)

        # add noise to users
        user_noise = torch.randn_like(user_emb)
        item_noise = torch.randn_like(item_emb)
        user_noise_emb = diff_model.forward_process(user_emb, ts, user_noise)
        item_noise_emb = diff_model.forward_process(item_emb, ts, item_noise)
        return user_noise_emb, item_noise_emb, ts, pt

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
        #return ssl_loss1

    def social_cas_ssl(self,social_seq_emb,cas_seq_emb):
        social_seq_emb=self.social_mlp(social_seq_emb)
        cas_seq_emb=self.cas_mlp(cas_seq_emb)
        ssl=self.calc_ssl_sim(social_seq_emb,cas_seq_emb,self.args.tau)
        return ssl

    def apply_noise1(self, user_emb, diff_model):
        # cat_emb shape: (batch_size*3, emb_size)
        emb_size = user_emb.shape[0]
        #batch_size, seq_len, emb_size = seq_size
        ts, pt = diff_model.sample_timesteps(emb_size, 'uniform')
        # ts_ = torch.tensor([self.config['steps'] - 1] * cat_emb.shape[0]).to(cat_emb.device)

        # add noise to users
        user_noise = torch.randn_like(user_emb)

        user_noise_emb = diff_model.forward_process(user_emb, ts, user_noise)

        return user_noise_emb, ts, pt

    def apply_noise2(self, user_emb, diff_model,seq_size):
        # cat_emb shape: (batch_size*3, emb_size)
        #emb_size = user_emb.shape[0]
        batch_size, seq_len, emb_size = seq_size
        # ts, pt = diff_model.sample_timesteps(emb_size, 'uniform')
        ts, pt = diff_model.sample_timesteps(batch_size, 'uniform')
        ts_expanded = ts.unsqueeze(1).repeat(1, seq_len).view(-1)
        # ts_ = torch.tensor([self.config['steps'] - 1] * cat_emb.shape[0]).to(cat_emb.device)

        # add noise to users
        user_noise = torch.randn_like(user_emb)

        user_noise_emb = diff_model.forward_process(user_emb,ts_expanded, user_noise)

        return user_noise_emb, ts_expanded, pt

    def time_diff_emb(self,cas_time):
        cas_timedifference = torch.diff(cas_time, dim=1)
        batch_size=cas_time.size(0)
        zero_padding = torch.zeros(batch_size, 1).to(cas_time.device)
        cas_timedifference_padded = torch.cat((zero_padding, cas_timedifference), dim=1)
        time_diff_embedding = self.time_diff_encoder(cas_timedifference_padded)
        return time_diff_embedding



import torch
import torch.nn as nn
import torch.nn.init as init
from .TransformerBlock import MultiheadAttention,TemporalSelfAttention
from .TransformerBlock import SelfAttention
from .RNN import RNN
from .NeuralNetwork import NeuralNetwork
import torch.nn.functional as F
from .HeteGT import *
import torch_geometric.utils as utils


class Attention(nn.Module):

    def __init__(self, in_features, hidden_size):   # 300*300
        super(Attention, self).__init__()
        self.linear1 = nn.Linear(in_features*2, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.reset_parameters()


    def reset_parameters(self):
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def forward(self, Q, V, mask = None):
        '''
        :param Q: (batch_size, d) 16*300
        :param V: (batch_size, hist_len, d)  16*30*300
        :return: (batch_size, d)
        '''
        # Q = Q.unsqueeze(dim=1).expand(V.size())  # 16*30*300
        fusion = torch.cat([Q, V], dim=-1)  # 16*30*600

        fc1 = self.activation(self.linear1(fusion))  # 16*30*300
        score = self.linear2(fc1)  # 16*30*1

        if mask is not None:
            mask = mask.unsqueeze(dim=-1)  # 16*30*1
            score = score.masked_fill(mask, -2 ** 32 + 1)

        alpha = F.softmax(score, dim=1)
        alpha = self.dropout(alpha)
        att = (alpha * V).sum(dim=1)  # 16*300
        return att

class DHGNN(NeuralNetwork):

    def __init__(self, config,  graph1, graph2, graph3):
        super(DHGNN, self).__init__()
        self.config = config
        embedding_weights = config['embedding_weights']
        V, D = embedding_weights.shape
        dropout_rate = config['dropout']

        self.node_init_tweet_feat = []
        self.graph1 = graph1
        self.graph2 = graph2
        self.graph3 = graph3
        self.num_relations = len(graph1.get_meta_graph())
        self.num_types = len(graph1.get_types())
        self.num_nodes = len(graph1.node_type)
        self.n_hid = config['n_hid']
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.num_layers = config['num_layers']
        self.n_out = config['n_out']
        self.rnn_name = config['rnn']
        self.adapt_ws = nn.ModuleList()
        self.gcs = nn.ModuleList()
        self.training = True    # dropout 邻接矩阵学习
        self.n_node = config['n_node']

        self.word_embedding = nn.Embedding(V, D, padding_idx=0, _weight=torch.from_numpy(embedding_weights))
        self.user_tweet_embedding = nn.Embedding(self.num_nodes-self.n_node, 400, padding_idx=0)  # twitter15:1490 twitter16:818 weibo:4664
        self.linear_graph_tweet_encode = nn.Linear(300, 400)

        self.linear_fuse = nn.Linear(600, 1)

        # HeteGT
        # for t in range(self.num_types):    # 节点类型线性层
        #     self.adapt_ws.append(nn.Linear(300, self.n_hid))
        for l in range(self.n_layers - 1):
            self.gcs.append(HeteGTConv(self.n_hid, self.n_hid, self.num_types, self.num_relations, self.n_heads, dropout_rate, use_norm = False).to(self.device))  # 前三层归一化
        self.gcs.append(HeteGTConv(self.n_hid, self.n_hid, self.num_types, self.num_relations, self.n_heads, dropout_rate, use_norm = False).to(self.device))    # 最后一层归一化
        #Dynamic_attention
        self.rnn = RNN(self.n_hid, self.n_hid, self.n_out, self.num_layers, self.rnn_name)
        #Attention
        self.mh_attention = MultiheadAttention(input_size=300, output_size=300, num_heads=2, is_layer_norm=False, attn_dropout=0.0)
        self.temporal_mh_attention = TemporalSelfAttention(input_size=300, output_size=300, num_heads=8, is_layer_norm=False, attn_dropout=0.0)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.fc_out = nn.Sequential(
            nn.Linear(600, 300),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(300, config['num_classes'])
        )
        self.reset_parameters()
        print(self)

    def reset_parameters(self):
        init.xavier_normal_(self.user_tweet_embedding.weight)
        init.xavier_normal_(self.linear_fuse.weight)
        for name, param in self.fc_out.named_parameters():
            if name.__contains__("weight"):
                init.xavier_normal_(param)

    def local_attention_network(self, X_source, X_replies, mask):
        '''
        Args:
            X_source: (batchSize,wordLen,word2vec) 16*50*300
            X_replies: (batchSize,replies,wordLen,word2vec) 16*30*50*300
            mask:
        Returns:
        '''
        replies=[]
        for replie in torch.chunk(X_replies, 30, dim=1):   # 将三十条相关评论拆开
            replies.append(replie.squeeze())  # 16*50*300
        '''
        词级注意力
        '''
        # 源推文多头自注意力
        word_selfAttention=self.mh_attention(X_source, X_source, X_source)  # 16*50*300
        post_source = torch.mean(word_selfAttention, 1)  # 平均池化 16*300

        # 源推文指导评论的多头交叉注意力
        word_crossAttention=[]
        for i in range(len(replies)):
            word_crossAttention.append(self.mh_attention(X_source, replies[i], replies[i]))  # 16*50*300

        post_replies = []
        for i in range(len(word_crossAttention)):
            # 最大池化
            post_replie = word_crossAttention[i].permute(0, 2, 1).contiguous()  # 16*300*50
            post_replie = F.adaptive_max_pool1d(post_replie, 1).permute(0, 2, 1).contiguous()  # 16*1*300 最大池化，转为句子级

            post_replies.append(post_replie)
        post_replies_self = torch.cat(post_replies, dim=1)  # 16*30*300

        '''
                句子级注意力
        '''
        post_source_cross = post_source.unsqueeze(dim=1).expand(post_replies_self.size())  # 16*30*300
        post_source_cross_self = self.mh_attention(post_source_cross, post_source_cross, post_source_cross)  # 16*30*300
        post_source_cross_self = torch.mean(post_source_cross_self, 1)  # 平均池化
        X_replie_cross = self.mh_attention(post_source_cross, post_replies_self, post_replies_self, mask)  # 16*30*300
        X_replie_cross = torch.mean(X_replie_cross, 1)  # 平均池化 16*300
        X_fuse = torch.cat([post_source, X_replie_cross], dim=-1)  # 16*600
        alpha = torch.sigmoid(self.linear_fuse(X_fuse))
        X_local = alpha * post_source + (1 - alpha) * X_replie_cross  # 16*300
        return X_local

    def global_tweet_encoding(self, graph):
        X_all_content = torch.LongTensor(graph.X_all_content).to(self.device)
        self.node_init_tweet_feat = self.word_embedding(X_all_content).to(torch.float32)  # 1490*50*300
        self.node_init_tweet_feat = torch.mean(self.node_init_tweet_feat, 1)   # 平均池化 1490*300
        self.node_init_tweet_feat = self.linear_graph_tweet_encode(self.node_init_tweet_feat)  # 1490*400

    def global_graph_encoding(self, X_tid, graph):
        self.global_tweet_encoding(graph)
        node_init_user_feat = self.user_tweet_embedding.weight
        node_init_feat = torch.cat([self.node_init_tweet_feat, node_init_user_feat], dim=0)
        node_init_feat = self.dropout(node_init_feat)

        edge_index = graph.edge_index
        edge_weight = graph.edges_weight
        node_type = graph.node_type
        # edge_type = graph.edge_type
        edge_types = []

        edge_index, _ = utils.dropout_adj(edge_index, p=0.6, training=self.training)
        size = edge_index.shape
        for i in range(size[1]):
            edge_types.append(0)  # 目前只考虑发布
        edge_type = torch.Tensor(edge_types)

        for gc in self.gcs:
            meta_xs = gc(node_init_feat.to(self.device), node_type.to(self.device), edge_index.to(self.device), edge_type.to(self.device))
        return meta_xs

    def dynamic_attention(self,X_global_splice1,X_global_splice2,X_global_splice3):
        X_rnn_input = torch.cat([X_global_splice1.unsqueeze(dim=1), X_global_splice2.unsqueeze(dim=1), X_global_splice3.unsqueeze(dim=1)], dim=1)
        x, hidden = self.rnn(X_rnn_input)
        x_selfAttention = self.mh_attention(x, x, x)  # 16*3*300
        x_mean = torch.mean(x_selfAttention, 1)  # 平均池化 16*300
        return x_mean

    def dynamic_attention_layer(self, X_tid, X_global_splice1, X_global_splice2, X_global_splice3):
        out1, hidden = self.rnn(X_global_splice1.unsqueeze(dim=1))
        out2, hidden = self.rnn(X_global_splice2.unsqueeze(dim=1))
        out3, hidden = self.rnn(X_global_splice3.unsqueeze(dim=1))
        tid_out = torch.cat([out1[X_tid],out2[X_tid],out3[X_tid]],dim=1)  # 16*3*300
        mask = torch.tril(torch.ones(3, 3), diagonal=-1).bool().unsqueeze(dim=0).expand(tid_out.size(0),3,3).to(self.device)
        x_selfAttention = self.temporal_mh_attention(tid_out, tid_out, tid_out, mask)  # 16*3*300
        output = []  # 取最后时间片
        for out in torch.chunk(x_selfAttention, 3, dim=1):
            output.append(out.squeeze())
        # x = output[2]
        x = torch.mean(x_selfAttention, 1)  # 平均池化 16*300
        return x

    def forward(self, X_tid, X_source, X_replies):
        mask = ((X_replies != 0).sum(dim=-1) == 0)  # 将无评论设为true 16*30
        X_source = self.word_embedding(X_source).to(torch.float32)  # (N*C, W, D) 16*50*300
        X_replies = self.word_embedding(X_replies).to(torch.float32)  # 16*30*50*300s

        X_local = self.local_attention_network(X_source, X_replies, mask)

        X_global_splice1 = self.global_graph_encoding(X_tid, self.graph1)  # 16*400
        X_global_splice2 = self.global_graph_encoding(X_tid, self.graph2)
        X_global_splice3 = self.global_graph_encoding(X_tid, self.graph3)

        # X_global = self.dynamic_attention(X_global_splice1, X_global_splice2, X_global_splice3)
        X_global = self.dynamic_attention_layer(X_tid, X_global_splice1, X_global_splice2, X_global_splice3)

        X_feat = torch.cat([X_local,X_global], dim=-1)
        X_feat = self.dropout(X_feat)

        output = self.fc_out(X_feat)
        return output

import torch.nn as nn
from .rum_model import RUM

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, rnn_name):
        '''
        :param intput_size: 输入维度
        :param hidden_size: 隐藏层维度
        :param output_size: 输出维度
        :param num_layer: 模型层数

        '''
        super(RNN, self).__init__()
        self.rnn_name=rnn_name

        '''
        batch_first=True: 输入数据的形式为[bs,seq,feature]
        bidirectional=True: 双向
        '''
        self.GRU_layer = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.RUM = RUM(input_size, hidden_size)
        self.output_linear = nn.Linear(hidden_size*2, output_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.hidden = None


    def forward(self, x):
        if self.rnn_name=='gru':
            # x[seq_len, batch, input_size]
            self.GRU_layer.flatten_parameters()
            x, self.hidden = self.GRU_layer(x)
            x = self.output_linear(x)  # x:16*3*300  hidden:2*16*400
        elif self.rnn_name=='rum':
            x, self.hidden = self.RUM(x)  # x:16*3*400  hidden:16*400
            x = self.linear(x)
        return x, self.hidden

import torch.nn as nn
from transformers import BertModel
# from config import *
from torchcrf import CRF
# from utils import set_logger
# from vocabulary import Vocabulary

'''
注意：如果数据量少 训练效果不好 则可以尝试让LSTM的层数小一点（收敛慢的主要原因） 或替换为GRU GRU通常也有较好的性能并且参数更少，有时可以更快地收敛。
'''
class BertBiLSTMCRF(nn.Module):

    def __init__(self, bert_model, lstm_dim, num_labels, dropout_probability=0):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob) #bert层的dropout  下面lstm可以直接加dropout参数

        # 如果设定 dropout 参数，那么 num_layers 必须大于 1。如果您希望保持 LSTM 的层数为 1，那么应将 dropout 参数设置为 0
        self.bilstm = nn.LSTM(input_size=self.bert.config.hidden_size,
                              hidden_size=lstm_dim,
                              num_layers=1,
                              dropout=dropout_probability,
                              bidirectional=True,
                              batch_first=True)
        self.classifier = nn.Linear(lstm_dim * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        '''
        Args:
            input_ids: 填充后的id  [batch_size, max_len]
            attention_mask:
            labels:
        '''
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 经过self.bert后 input_ids才变为768维
        sequence_output = bert_output.last_hidden_state #或bert_output[0]
        sequence_output = self.dropout(sequence_output)
        lstm_output, _ = self.bilstm(sequence_output)
        emissions = self.classifier(lstm_output)
        #如果传入真实labels则计算损失 否则进行预测
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask)  # 注意：可能需要调整为bool类型，取决于PyTorch版本
            return loss
        else:
            predictions = self.crf.decode(emissions, mask=attention_mask)  # 注意：同上
            return predictions


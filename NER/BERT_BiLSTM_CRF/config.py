'''
配置文件  定义常量和超参数
# 作者：刘泽宇
# 创建日期：2023年11月7日
'''

# 训练数据集
TRAIN_SAMPLE_PATH = '../data/dataset_20240420/train.csv'
# 测试数据集
TEST_SAMPLE_PATH = '../data/dataset_20240420/dev.csv'
# 验证数据集
DEV_SAMPLE_PATH = '../data/dataset_20240420/dev.csv'

#标签表路径 bert自带词表
LABEL_PATH = '../data/output_dataset_20240420/label.txt'

# 是否对整个模型进行全参数微调
full_fine_tuning = True
bert_model = '../../pretrained_bert_models/pubmedbert-base-uncased/'
# roberta_model = 'pretrained_bert_models/pubmedbert-base-uncased/'
MAX_POSITION_EMBEDDINGS = 512

# 设置pad_id 和 label_o_id
WORD_PAD_ID = 0 #bert词表中pad=0  是词表第一个  可以在bert的vocab中查看
LABEL_O_ID = 0 # 跟输出的词表中保持一致  O为0  在代码中保证了“O”在第一个

'''
bert有自己的词表 不用自己构建  只需构建标签表
'''

# hyper-parameter
# 学习率
LR = 5e-5
weight_decay = 1e-4
clip_grad = 5 #梯度裁剪的最大值

# LSTM输出的隐层的大小
HIDDEN_SIZE = 128

BATCH_SIZE = 10
# 迭代轮数
EPOCH_NUMS = 50
# 训练好的模型存放地址
MODEL_DIR = '../model/ckpt_dataset_20240420/'
# 日志存放地址
LOG_PATH = '../logger/dataset_20240420_train.log'

DROPOUT_PROBABILITY = 0
MIN_EPOCH_NUM = 30
PATIENCE = 0.0001
PATIENCE_NUM = 7

import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu'
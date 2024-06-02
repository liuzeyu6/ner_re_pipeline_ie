import torch
import os
'''
配置文件  定义常量和超参数
# author：liuzeyu
# data：2024.05.30
'''
# 日志存放地址
# LOG_PATH = 'D:/医学数据/pipeline_ie/logger/Pipeline.log'
LOG_PATH = 'logger/pipeline.log'
# 训练数据集
TRAIN_SAMPLE_PATH = 'NER/data/dataset_20240420/train.csv'
# 测试数据集
TEST_SAMPLE_PATH = 'NER/data/dataset_20240420/dev.csv'
# 验证数据集
DEV_SAMPLE_PATH = 'NER/data/dataset_20240420/dev.csv'

#标签表路径 bert自带词表
LABEL_PATH = 'NER/data/output_dataset_20240420/label.txt'


# 是否对整个模型进行全参数微调
full_fine_tuning = True
bert_model_path = 'pretrained_bert_models/pubmedbert-base-uncased/'
# roberta_model = 'pretrained_bert_models/pubmedbert-base-uncased/'
MAX_POSITION_EMBEDDINGS = 512

# 设置pad_id 和 label_o_id
WORD_PAD_ID = 0  #bert词表中pad=0  是词表第一个  可以在bert的vocab中查看
LABEL_O_ID = 0  # 跟输出的词表中保持一致  O为0  在代码中保证了“O”在第一个

'''
bert有自己的词表 不用自己构建  只需构建标签表
'''

# hyper-parameter
# 学习率
LR = 5e-5
weight_decay = 1e-4
clip_grad = 5  #梯度裁剪的最大值

# LSTM输出的隐层的大小
HIDDEN_SIZE = 128

BATCH_SIZE = 10
# 迭代轮数
EPOCH_NUMS = 50
# 训练好的模型存放地址
MODEL_DIR = 'NER/model/ckpt_dataset_20240420/'

# 训练好的NER模型
# ner_model_path = "../model/ckpt_dataset_20240420/ner_model_25.pth"
ner_model_path = "NER/model/ckpt_dataset_20240420/ner_model.pth"

DROPOUT_PROBABILITY = 0
MIN_EPOCH_NUM = 30
PATIENCE = 0.0001
PATIENCE_NUM = 7

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu'


# RE配置文件
re_root_path = './RE'
re_model_name = 'lzy_pipeline_bert_entity'
re_model_path = os.path.join(re_root_path, 'ckpt/' + re_model_name + '.pth.tar')
res_path = 'pred_res/triple.csv'


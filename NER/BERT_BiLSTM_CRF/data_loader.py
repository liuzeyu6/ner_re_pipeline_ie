from transformers import BertTokenizer
from config import *
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from vocabulary import Vocabulary

'''
定义数据集 
- type 参数，这个类是训练和测试公用的，所以定义一个参数来区分加载哪个文件,默认值是训练集。
- tokenizer:bert的分词器 
- base_len 参数，用来定义句子的参考长度默认值是50（切分句子），特殊情况再稍做处理。
'''
'''
tokenizer.encode_plus和tokenizer.encode的区别：
两者的主要区别在于它们返回的信息量。
（1）tokenizer.encode:
返回一个整数列表，这些整数代表文本中每个token的ID。
自动添加特殊tokens（比如对于BERT模型是[CLS]和[SEP]），可以通过设置add_special_tokens=False不加特殊符号  CLS分类任务时有用
通常用于简单的任务，当你只需要token IDs，并且不需要额外的信息时。
（2）tokenizer.encode_plus:
返回一个字典，包含了更多的信息，不仅仅是token IDs。
字典中还可能包含token_type_ids、attention_mask、overflowing_tokens等字段，具体取决于调用时传入的参数。
token_type_ids用于区分多个句子（对于BERT等模型，句子对的任务很有用）。
attention_mask告诉模型哪些tokens是真实的，哪些是为了满足最大序列长度要求而添加的填充tokens。
还可以控制是否返回特殊tokens、填充策略、截断策略等。
适用于需要这些额外信息的复杂任务。
（3)tokenizer()
在最新的版本中，encode_plus的功能通常可以通过调用tokenizer对象本身来实现，这样的调用通常等价于encode_plus的调用，且更为简洁。
'''
# 数据集类
class NERDataset(Dataset):
    def __init__(self, tokenizer, vocab, mode='train', base_len=50):
        super().__init__()
        self.base_len = base_len
        if mode == 'train':
            sample_path = TRAIN_SAMPLE_PATH
        elif mode == 'test':
            sample_path = TEST_SAMPLE_PATH
        elif mode == 'dev':
            sample_path = DEV_SAMPLE_PATH
        else:
            raise ValueError("mode must be one of train, dev, or test")
        self.data = pd.read_csv(sample_path)
        # self.label2id = {label: id for id, label in enumerate(self.data['label'].unique())}
        # self.id2label = {id: label for label, id in self.label2id.items()}
        self.id2label, self.label2id = vocab.get_label()
        self.tokenizer = tokenizer
        self.get_points()

    def get_points(self):
        '''
        采用等长切分，每隔base_len个字切分一次。
        但有一种情况需要处理，切点上是非O标签，则需要将切点往后移动，直达O标签为止（避免切到实体上）。
        Returns:self.points=[0，50，102，140]
        '''
        self.points = [0]
        i = 0
        while True:
            # 判断是否为结尾 最后一次切分
            if i + self.base_len >= len(self.data):
                self.points.append(len(self.data))
                break
            # 判断切分位置是否为O标签 不是则继续后移
            if self.data.loc[i + self.base_len, 'label'] == 'O':
                i += self.base_len
                self.points.append(i)
            else:
                i += 1

    def __len__(self):
        # 计算共有多少个句子的方法  points列表长度-1
        return len(self.points) - 1

    def __getitem__(self, index):
        # 此时的__getitem__方法还有问题，即拿到的单个句子向量长度不都是50，转tensor会报错。
        # 该方法将长文本切分成了单句，但每个句子长度又不完全一样。
        # 取到单个句子的方法  index即points的索引
        df = self.data[self.points[index]:self.points[index + 1]]
        # 先对df切片 拿到第index个句子的word和label并转为向量
        label_o_id = self.label2id['O']
        # 列表推导式 ：先循环word取到w，get方法若w存在则取到其id，否则是未知id
        # input_ids = [self.word2id.get(w, word_unk_id) for w in df['word']]

        # 加bert只需修改input_ids的获取方式  改为tokenizer的encode方法  不需要填充 也不需要特殊符号cls 和 sep因为已经是一个一个字的输入
        # input_ids = self.tokenizer.encode(list(df['word']), add_special_tokens=False) #加list是怕英文单词转为一个id 而原文一个字母对应一个tag
        input_ids = self.tokenizer.encode_plus(list(df['word']), add_special_tokens=False)[
            'input_ids']  # 加list是怕英文单词转为一个id 而原文一个字母对应一个tag

        # 在NER（命名实体识别）任务中，通常不需要使用CLS（分类）标记，因为NER任务通常被视为序列标记任务，而不是分类任务。CLS标记在BERT等预训练模型中通常用于句子级别的任务，如文本分类，但在NER任务中没有直接的用处。

        # input_ids = [self.tokenizer.cls_token_id] + input_ids  # CLS SEP PAD label都为O  NER任务可以不要SEP ， CLS待定 先加上
        label_ids = [self.label2id.get(l, label_o_id) for l in df['label']]
        # label_ids = [label_o_id]+label_ids
        # print(input_ids)
        # print(len(input_ids))
        # print(label_ids)
        # print(len(label_ids))
        # exit()
        return input_ids[:MAX_POSITION_EMBEDDINGS], label_ids[:MAX_POSITION_EMBEDDINGS]  # MAX_POSITION_EMBEDDINGS=512是base版本最大处理长度


# 在批处理时，要求每个batch的句子长度必须一致，所以我们需要填充 PAD，来保证句子每个batch的句子长度一样。
# 需要在 DataLoader 批量加载数据阶段，填充 PAD 来保证批数据长度一致，且需要记录 MASK（在后续模型 CRF 阶段计算损失时，可以通过MASK，将填充的 PAD 数值忽略掉，以消除填充值 PAD 的影响。）。
def collate_fn(batch):
    # collate_fn是一个回调函数：填充PAD和设置MASK都是在这个函数中实现（类似web的中间件，从数据进来到返回中间对数据的处理）回调函数只需要传一个函数名。
    # key定义排序规则：len（x[0]）表示以元组第一个元素长度排序。倒序从大到小
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    # 倒序后第一个即最长句子
    max_len = len(batch[0][0])
    input_ids = []
    label_ids = []
    input_mask = []
    for item in batch:
        # item是input_ids和label_ids的元组
        # 最长的句子减去该句长度即需要填充的长度
        pad_len = max_len - len(item[0])
        # word填充PAD
        input_ids.append(item[0] + [WORD_PAD_ID] * pad_len)
        # label填充O
        label_ids.append(item[1] + [LABEL_O_ID] * pad_len)
        # mask掉填充的字符，即原先的句子长度填1，后面我们填充的为0；
        input_mask.append([1] * len(item[0]) + [0] * pad_len)

    # 最后把数据转为模型需要的tensor形式。mask转为byte或bool型是CRF的要求,取决于pytorch版本
    return torch.tensor(input_ids), torch.tensor(label_ids), torch.tensor(input_mask).bool()


if __name__ == '__main__':
    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    # 加载标签表
    vocab = Vocabulary()
    # 创建数据集
    train_dataset = NERDataset(mode='train', vocab=vocab, tokenizer=tokenizer, base_len=15)
    dev_dataset = NERDataset(mode='dev',vocab=vocab, tokenizer=tokenizer, base_len=15)
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=6, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=6, collate_fn=collate_fn)

    # 迭代DataLoader对象
    for data in train_loader:
        # 在每次迭代中处理数据\
        print(data)
        exit()



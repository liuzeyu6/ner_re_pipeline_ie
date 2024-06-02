from transformers import BertTokenizerFast
import torch

class NERPredict():
    def __init__(self, vocab, ner_model, pretrained_model_path, device='cpu'):
        # self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        self.device = device
        #bert不需要id2word  有自己的词表  convert_ids_to_tokens
        self.id2label, self.label2id = vocab.get_label()
        self.tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_path)
        self.model = ner_model
        self.model.eval()

    def tokenize_sentence(self, text):
        '''
        tokenizer.encode:
            返回一个整数列表，这些整数代表文本中每个token的ID。
            自动添加特殊tokens（比如对于BERT模型是[CLS]和[SEP]）。
            通常用于简单的任务，当你只需要token IDs，并且不需要额外的信息时。
        tokenizer.encode_plus:
            返回一个字典，包含了更多的信息，不仅仅是token IDs。
            字典中还可能包含token_type_ids、attention_mask、overflowing_tokens等字段，具体取决于调用时传入的参数。
            token_type_ids用于区分多个句子（对于BERT等模型，句子对的任务很有用）。
            attention_mask告诉模型哪些tokens是真实的，哪些是为了满足最大序列长度要求而添加的填充tokens。
            还可以控制是否返回特殊tokens、填充策略、截断策略等。
            适用于需要这些额外信息的复杂任务。
        tokenizer()
            在最新的版本中，encode_plus的功能通常可以通过调用tokenizer对象本身来实现，这样的调用通常等价于encode_plus的调用，且更为简洁。
            tokenizer("Hello, how are you?", add_special_tokens=True, max_length=512, padding='max_length', return_attention_mask=True)
        tokenizer.batch_encode_plus
        对一组文本进行处理，返回输入 ID、注意掩码等信息，以便这些文本可以被传递给模型进行推理。输入是一个字符串列表或字符串对的列表（即，单个文本或文本对的列表）。
        padding (默认: True):
            True 或 'longest': 将所有序列填充到批处理中的最长序列的长度。
            'max_length': 将所有序列填充到 max_length。
            False: 不进行填充。
        truncation (默认: True):
            True: 将序列截断到最大长度（由模型或 max_length 参数指定）。
            False: 不进行截断。
            'longest_first': 优先截断最长的序列。
            'only_first' 和 'only_second': 仅截断第一或第二个序列（仅对文本对适用）。

        max_length (默认: None):
            指定序列的最大长度。如果没有设置，则使用模型的最大长度。
        verbose (默认: True):
            是否打印处理中的警告信息。

        '''
        # 如果只是单句子预测 不需要填充，但bert这样的模型最多接受512个token 需要truncation截断
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=False,
            max_length=512,
            return_attention_mask=True,
            # padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        offsets_mapping = encoding['offset_mapping'].to(self.device)
        return input_ids, attention_mask.bool(), offsets_mapping

    def predict(self, text):
        """
        注意：英文预测和中文不一样 ，使用bert分词会有 一个word被拆分为多个子词的情况：##  需要用tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])获取实体
        Args:
            text: 输入文本

        Returns:

        """
        # 对句子进行分句并且tokenize
        input_ids, attention_mask, offset_mapping = self.tokenize_sentence(text)
        offset_mapping = offset_mapping[0]

        with torch.no_grad():
            y_pred = self.model(input_ids, attention_mask)[0]

        labels = [self.id2label[l] for l in y_pred]
        # 获取tokenizer生成的tokens
        # tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        # 用start end定位entity在tokens列表中的位置  然后用根据start和end找到对应原文中的offset_mapping
        i = 0
        res = []
        while i < len(labels):
            # 如果不是O  则说明实体chunk开始
            if labels[i] != 'O':
                if labels[i].startswith('B'):
                    prefix, name = labels[i].split('-')
                    start = end = i
                    i += 1
                    while i < len(labels) and (labels[i] == ('I-' + name) or labels[i] == ('E-' + name)):
                        end = i
                        i += 1
                    # 获取原始文本中的起始和结束位置
                    start_offset = offset_mapping[start][0].item()
                    end_offset = offset_mapping[end][1].item()
                    # 要使用offset在原文本中找 而不是在token中找  token会分词##
                    entity = text[start_offset:end_offset]
                    label = name
                    res.append({
                        "entity": entity,
                        "label": label,
                        "pos_start": start_offset,
                        "pos_end": end_offset
                    })
                else:
                    i += 1
            else:
                i += 1
        return res

    def tokenize_sentences(self, texts):
        # 支持批量处理 max_length和truncation一起使用
        encoding = self.tokenizer.batch_encode_plus(
            texts,
            return_attention_mask=True,
            add_special_tokens=False,
            # padding='max_length',  # 一律补零，直到max_length长度
            max_length=512,
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        offsets_mapping = encoding['offset_mapping'].to(self.device)
        return input_ids, attention_mask.bool(), offsets_mapping

    def predict_batch(self, texts):
        """批量预测——只保留B开头I或E结尾的"""
        input_ids, attention_mask, offset_mapping = self.tokenize_sentences(texts)
        with torch.no_grad():
            y_pred = self.model(input_ids, attention_mask)
        batch_predictions = []
        for idx, text in enumerate(texts):
            labels = [self.id2label[l] for l in y_pred[idx]]
            i = 0
            res = []
            while i < len(labels):
                if labels[i] != 'O':  # 非'O'标签表示实体的开始
                    if labels[i].startswith('B'):
                        prefix, name = labels[i].split('-')
                        start = end = i
                        i += 1
                        # 找到实体的结尾  注意：可能出现咳嗽咳痰 BEBE这种
                        while i < len(labels) and (labels[i] == ('I-' + name) or labels[i] == ('E-' + name)):
                            end = i
                            i += 1
                        # 获取原始文本中的起始和结束位置
                        start_offset = offset_mapping[idx][start][0].item()
                        end_offset = offset_mapping[idx][end][1].item()
                        # 要使用offset在原文本中找 而不是在token中找  token会分词##
                        entity = text[start_offset:end_offset]
                        label = name
                        res.append({
                            "entity": entity,
                            "label": label,
                            "pos_start": start_offset,
                            "pos_end": end_offset
                        })
                    else:
                        i += 1
                else:
                    i += 1

            batch_predictions.append(res)
        return batch_predictions
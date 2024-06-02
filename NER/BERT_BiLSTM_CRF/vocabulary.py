import os
import pandas as pd
import logging

class Vocabulary:
    """
    构建标签表 bert不需要自己构建词表  只需根据训练集构建标签表
    """

    def __init__(self, train_path, label_path):
        self.id2label = None
        self.label2id = None
        self.train_data_path = train_path
        self.label_path = label_path


    def __len__(self):
        return len(self.label2id) if self.label2id is not None else 0

    def label_size(self):
        return len(self.label2id) if self.label2id is not None else 0

    def load_data(self):
        """
        Load data from training dataset. 模拟真实情景——只用训练集数据生成词典和标签表
        """
        try:
            df = pd.read_csv(self.train_data_path)
            return df
        except Exception as e:
            logging.error(f"Error loading data from {self.train_data_path}: {e}")
            return None

    def generate_label(self, df):
        """
        Generate labels from a DataFrame.
        """
        if df is not None:
            # 生成标签列表并转换为字典
            label_list = df['label'].value_counts().keys().tolist()
            label_dict = {v: k for k, v in enumerate(label_list)}
            # 使用断言检查键 "O" 是否存在于字典中，如果不存在则触发 AssertionError
            assert 'O' in label_dict, "Error: Key 'O' not found in dictionary"
            # 确保 "O" 标签位于标签字典的第一个位置
            # 如果 "O" 在标签字典中，将其移动到第一个位置
            label_dict = {"O": 0, **{k: v for k, v in label_dict.items() if k != "O"}}
            # 更新 id2label 和 label2id 属性
            self.id2label, self.label2id = list(label_dict.keys()), label_dict
            logging.info("Label vocab generated successfully.")

    def save_vocab_label(self):
        """
        Save label to files.
        """
        if self.label2id is not None:
            if not os.path.exists(os.path.dirname(self.label_path)):
                os.makedirs(os.path.dirname(self.label_path)) #os.makedirs 接受文件夹路径，而不是文件路径。所以要os.path.dirname获取目录

            pd.DataFrame(list(self.label2id.items())).to_csv(self.label_path, header=None, index=None)
            logging.info("Label vocab saved to files successfully.")
        else:
            logging.error("Label vocab not generated yet.")

    def get_label(self):
        '''
        读取LABEL_PATH的label表
        Returns:id2label和label2id以及标签“O”的id
        '''
        df = pd.read_csv(self.label_path, names=['label', 'id'])
        self.id2label = list(df['label'])
        self.label2id = dict(df.values)
        return self.id2label, self.label2id


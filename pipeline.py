import pandas as pd
from NER.BERT_BiLSTM_CRF.predict import NERPredict
from NER.BERT_BiLSTM_CRF.vocabulary import Vocabulary
from NER.BERT_BiLSTM_CRF.ner_model import BertBiLSTMCRF
from RE.inference.inference import get_model
import config
import torch
from logger import *

def extract_relations(entity_list, sentence, re_model, thr=0.5):
    triples = []
    for i, entity1 in enumerate(entity_list):
        for j, entity2 in enumerate(entity_list):
            if i != j:
                inputs = prepare_re_inputs(entity1, entity2, sentence)  # 根据具体的RE模型准备输入
                with torch.no_grad():
                    rel, score = re_model.infer(inputs)
                    if score < thr:
                        continue
                    triples.append({
                        "entity_h": entity1["entity"],
                        "rel": rel,
                        "rel_score": score,
                        "entity_t": entity2["entity"],
                    })
    return triples


def prepare_re_inputs(entity_h, entity_t, sentence):
    # 根据具体的RE模型准备输入
    re_input = {
        "text": sentence,
        "t": {"pos": [entity_t["pos_start"], entity_t["pos_end"]]},
        "h": {"pos": [entity_h["pos_start"], entity_h["pos_end"]]},
    }
    return re_input


if __name__ == '__main__':
    # 示例调用
    set_logger(config.LOG_PATH)
    logging.info(f"--------The device you are using is {config.DEVICE}--------")

    vocab = Vocabulary(train_path=config.TRAIN_SAMPLE_PATH, label_path=config.LABEL_PATH)
    vocab.get_label()
    logging.info(f"--------Loading NER model from {config.ner_model_path}--------")
    ner_model = BertBiLSTMCRF(bert_model=config.bert_model_path, lstm_dim=config.HIDDEN_SIZE,
                              num_labels=vocab.label_size()).to(config.DEVICE)

    ner_model.load_state_dict(torch.load(config.ner_model_path, map_location=config.DEVICE))
    ner_predictor = NERPredict(ner_model=ner_model, vocab=vocab, pretrained_model_path=config.bert_model_path, device=config.DEVICE)
    logging.info("--------NER model has been loaded successfully!--------")

    logging.info(f"--------Loading RE model from {config.re_model_path}--------")
    re_predictor = get_model(model_name=config.re_model_name, dataset='pipeline_data', device=config.DEVICE,
                             pretrained_path=config.bert_model_path, root_path=config.re_root_path,
                             re_model_path=config.re_model_path)
    logging.info("--------RE model has been loaded successfully!--------")

    # 提取实体
    text = "AB：Leonuri herba (I-mu-ts'ao, the Chinese motherwort) is an ancient Chinese traditional herb. TI：Enhancement of phenylephrine-induced contraction in the isolated rat aorta with endothelium by H2O-extract from an Oriental medicinal plant Leonuri herba."
    pred_entity_list = ner_predictor.predict(text)
    # print(pred_entity_list)
    # 提取关系
    triple_list = extract_relations(pred_entity_list, text, re_predictor, thr=0.7)
    # print(triple_list)
    # 写入 CSV 文件
    # 将数据转换为 DataFrame
    df = pd.DataFrame(triple_list)
    # 将 DataFrame 写入 CSV 文件
    df.to_csv(config.res_path, index=False, encoding='utf-8')

    logging.info(f'Inference result has been written to {config.res_path}')




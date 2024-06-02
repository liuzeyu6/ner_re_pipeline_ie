from predict import *


def extract_relations(entity_list, sentence):
    relations = []
    for i, entity1 in enumerate(entity_list):
        for j, entity2 in enumerate(entity_list):
            if i != j:
                inputs = prepare_re_inputs(entity1, entity2, sentence)  # 根据具体的RE模型准备输入
                print(inputs)
                exit()
                # with torch.no_grad():
                #     re_outputs = re_model(inputs)
                #
                # # 将RE模型的输出转换为关系类型
                # relation = postprocess_re_outputs(re_outputs)
                #
                # if relation:
                #     relations.append((entity1, relation, entity2))

    return relations


def prepare_re_inputs(entity_h, entity_t, sentence):
    # 根据具体的RE模型准备输入
    re_input = {
        "text": sentence,
        "t": {"pos": [entity_t["pos_start"], entity_t["pos_end"]]},
        "h": {"pos": [entity_h["pos_start"], entity_h["pos_end"]]},
    }
    return re_input


if __name__ == '__main__':
    set_logger(LOG_PATH)
    vocab = Vocabulary()

    ner_model_path = r"D:\医学数据\tianjin_data\NER_EN\BERT_BiLSTM_CRF_NER\model\ckpt_dataset_20240420\ner_model_25.pth"
    ner_predictor = Predict(load_model_path=ner_model_path, vocab=vocab)
    print("--------NER model has been loaded successfully!--------")

    txt = "AB：Leonuri herba (I-mu-ts'ao, the Chinese motherwort) is an ancient Chinese traditional herb. TI：Enhancement of phenylephrine-induced contraction in the isolated rat aorta with endothelium by H2O-extract from an Oriental medicinal plant Leonuri herba."
    pred_entity_list = ner_predictor.predict(txt)
    print(pred_entity_list)
    triple_list = extract_relations(pred_entity_list, txt)

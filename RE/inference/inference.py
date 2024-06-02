import json
import os
import numpy as np
import torch
from RE.opennre import encoder, model


def get_model(model_name, dataset, device, pretrained_path, root_path, re_model_path):
    ckpt = re_model_path
    if 'cnn' in model_name:
        word2id = json.load(open(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_word2id.json')))
        word2vec = np.load(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_mat.npy'))
        rel2id = json.load(open(os.path.join(root_path, f'dataset/{dataset}/rel2id.json')))
        sentence_encoder = encoder.CNNEncoder(token2id=word2id,
                                              max_length=512,
                                              word_size=50,
                                              position_size=5,
                                              hidden_size=230,
                                              blank_padding=True,
                                              kernel_size=3,
                                              padding_size=1,
                                              word2vec=word2vec,
                                              dropout=0.5)
        m = model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
        m.load_state_dict(torch.load(ckpt, map_location=device)['state_dict'])
        return m

    elif 'bert' in model_name:
        rel2id = json.load(open(os.path.join(root_path, f'dataset/{dataset}/rel2id.json')))
        if 'entity' in model_name:
            sentence_encoder = encoder.BERTEntityEncoder(
                max_length=512, pretrain_path=pretrained_path)
        else:
            sentence_encoder = encoder.BERTEncoder(
                max_length=512, pretrain_path=pretrained_path)
        m = model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
        m.load_state_dict(torch.load(ckpt, map_location=device)['state_dict'])
        return m
    else:
        raise NotImplementedError


if __name__ == '__main__':
    # 项目根目录
    default_root_path = '../'
    # 预训练模型路径
    pretrained_path = '../../pretrained_bert_models/pubmedbert-base-uncased'
    model = get_model(model_name='lzy_pipeline_bert_entity', dataset='pipeline_data', device='cuda',
                      pretrained_path=pretrained_path, root_path=default_root_path)
    result = model.infer({
                             "text": "AB：ETHNOPHARMACOLOGICAL RELEVANCE: Traditional Chinese medicine Leonurus japonicus Houtt. has a long history in the treatment of cardiovascular diseases. Stachydrine hydrochloride, the main bioactive ingredient extracted from Leonurus japonicus Houtt., has been shown to have cardioprotective effects. However, the underlying mechanisms of stachydrine hydrochloride haven't been comprehensively studied so far. AIM OF THE STUDY: The aim of this study was to investigate the protective role of stachydrine hydrochloride in heart failure and elucidate its possible mechanisms of action. MATERIALS AND METHODS: In vivo, transverse aorta constriction was carried out in C57BL/6J mice, and thereafter, 7.2 mg/kg telmisartan (a selective AT1R antagonist as positive control) and 12 mg/kg stachydrine hydrochloride was administered daily intragastrically for 4 weeks. Cardiac function was evaluated by assessing morphological changes as well as echocardiographic and haemodynamic parameters. In vitro, neonatal rat cardiomyocytes or adult mice cardiomyocytes were treated with stachydrine hydrochloride and challenged with phenylephrine (α-AR agonist). Ventricular myocytes were isolated from the hearts of C57BL/6J mice by Langendorff crossflow perfusion system. Intracellular calcium was measured by an ion imaging system. The length and movement of sarcomere were traced to evaluate the systolic and diastolic function of single myocardial cells. RESULTS: Stachydrine hydrochloride improved the cardiac function and calcium transient amplitudes, and inhibited the SR leakage and the amount of sparks in cardiac myocytes isolated from TAC mice. We also demonstrated that stachydrine hydrochloride could ameliorated phenylephrine-induced enhance in sarcomere contraction, calcium transients and calcium sparks. Moreover, our data shown that stachydrine hydrochloride blocked the hyper-phosphorylation of CaMKII, RyR2, PLN, and prevented the disassociation of FKBP12.6 from RyR2. CONCLUSION: Our results suggest that stachydrine hydrochloride exerts beneficial therapeutic effects against heart failure. These cardioprotective effects may be associated with the regulation of calcium handling by stachydrine hydrochloride through inhibiting the hyper-phosphorylation of CaMKII.",
                             "t": {"pos": [1425, 1441]},
                             "h": {"pos": [1383, 1414]}})  #     "relation": "in",
    print(result)

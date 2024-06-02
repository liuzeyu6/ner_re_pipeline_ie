from data_loader import *
from utils import *
from ner_model import *
from config import *
from train import train
from test import test
from transformers.optimization import get_cosine_schedule_with_warmup, AdamW
import warnings

warnings.filterwarnings('ignore')


def run():
    """train the model"""
    # set the logger
    set_logger(LOG_PATH)
    logging.info("device: {}".format(DEVICE))
    vocab = Vocabulary()
    data = vocab.load_data(TRAIN_SAMPLE_PATH)
    vocab.generate_label(data)
    vocab.save_vocab_label()
    logging.info("device: {}".format(DEVICE))

    # 加载tokenizer 这里用的是本地模型 不能用AutoTokenizer
    tokenizer = BertTokenizer.from_pretrained(
        bert_model)  # AutoTokenizer 是 transformers 库中的一个类，用于根据指定的预训练模型自动选择并加载正确的tokenizer类和预训练权重。这个类是非常有用的，可以根据你提供的模型名称（如 "bert-base-uncased-uncased"、"gpt2" 等）自动识别并实例化相应的tokenizer。
    # 创建数据集 build dataset
    # 创建数据集
    train_dataset = NERDataset(mode='train', tokenizer=tokenizer, vocab=vocab, base_len=50)
    dev_dataset = NERDataset(mode='dev', tokenizer=tokenizer, vocab=vocab, base_len=50)
    test_dataset = NERDataset(mode='test', tokenizer=tokenizer, vocab=vocab, base_len=50)

    train_size = len(train_dataset)
    dev_size = len(dev_dataset)
    test_size = len(test_dataset)

    logging.info(f"length of train_dataset:{train_size}")
    logging.info(f"length of dev_dataset:{dev_size}")
    logging.info(f"length of test_dataset:{test_size}")
    logging.info("--------Dataset Build!--------")

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    logging.info("--------Get Dataloader!--------")

    # Prepare model
    model = BertBiLSTMCRF(bert_model=bert_model, lstm_dim=HIDDEN_SIZE, num_labels=vocab.label_size()).to(DEVICE)
    logging.info(model)

    '''
    为模型的不同部分（如BERT层、LSTM层、分类器）设置不同的学习率和权重衰减策略。BERT层可能使用不同的学习率和权重衰减值，而LSTM层和分类器则使用另一套值。
    （1）全参数微调（Full Fine-Tuning）:
        如果 full_fine_tuning 为 True，则对整个模型的所有层进行微调。
        从模型的BERT层、BiLSTM层、分类器层和CRF层中提取参数。
        对于不同的层，设置不同的优化参数，包括学习率和权重衰减。
    （2）头部分类器微调（Fine-Tuning the Classifier Only）:
        如果 full_fine_tuning 为 False，则只对分类器层的参数进行微调。
        此时，忽略模型的其他层，只关注分类器层的参数。
    '''
    # Prepare optimizer 是否微调bert（使用不同学习率） 指定权重衰减的比率。weight_decay 权重衰减是一种正则化技术，主要用于防止模型过拟合
    if full_fine_tuning:
        # model.named_parameters(): [bert, bilstm, classifier, crf] classifier即全连接层
        bert_optimizer = list(model.bert.named_parameters())
        lstm_optimizer = list(model.bilstm.named_parameters())
        classifier_optimizer = list(model.classifier.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight'] #no_decay: 这个列表包含不应用权重衰减的参数类型。通常，对模型中的偏置项（bias）和LayerNorm层的参数不使用权重衰减，因为对这些参数应用权重衰减可能会对模型的学习过程产生负面影响。
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay,'lr':LR},
            {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0,'lr':LR},

            {'params': [p for n, p in lstm_optimizer if not any(nd in n for nd in no_decay)],
             'lr': LR * 5, 'weight_decay': weight_decay},
            {'params': [p for n, p in lstm_optimizer if any(nd in n for nd in no_decay)],
             'lr': LR * 5, 'weight_decay': 0.0},

            {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
             'lr': LR * 5, 'weight_decay': weight_decay},
            {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
             'lr': LR * 5, 'weight_decay': 0.0},

            {'params': model.crf.parameters(), 'lr': LR * 5}
        ]
    # only fine-tune the head classifier
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LR, correct_bias=False) #一些研究和实践表明，禁用偏差校正可能更合适。这可能是因为预训练模型已经在大规模数据集上进行了充分的训练，所以在微调阶段不再需要偏差校正。

    train_steps_per_epoch = train_size // BATCH_SIZE #计算每个epoch的训练步数 即iterations
    # get_cosine_schedule_with_warmup 是一个创建余弦退火调度器的函数，这种调度器会在初期的预热阶段逐渐增加学习率，然后根据余弦函数逐渐减少学习率。
    #-num_warmup_steps: 预热阶段的步数。在这里，它被设置为总训练轮数（config.epoch_num）的十分之一，乘以每个epoch的训练步数。即在初始10%的训练期间，学习率会逐渐增加。
    # -num_training_steps: 总的训练步数，等于总训练轮数乘以每个epoch的训练步数。
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=(EPOCH_NUMS // 10) * train_steps_per_epoch,
                                                num_training_steps=EPOCH_NUMS * train_steps_per_epoch)

    # Train the model
    logging.info("--------Start Training!--------")
    train(train_loader=train_loader, dev_loader=dev_loader, id2label=vocab.id2label, model=model, optimizer=optimizer,
          device=DEVICE, scheduler=scheduler)
    with torch.no_grad():
        # test on the final test set
        test(id2label=vocab.id2label, device=DEVICE, test_dataset=test_dataset)


if __name__ == '__main__':
    run()

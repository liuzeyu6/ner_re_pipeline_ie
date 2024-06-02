from prettytable import PrettyTable
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from data_loader import *
from utils import *
from ner_model import *
from config import *


def epoch_train(train_loader, model, optimizer, device, epoch, scheduler):
# def epoch_train(train_loader, model, optimizer, device, epoch):
    # set model to training mode
    model.train()
    # step number in one epoch: 336
    train_loss = 0.0
    for idx, batch_samples in enumerate(tqdm(train_loader)):
        input_ids, label_ids, input_mask = batch_samples
        input_ids = input_ids.to(device)
        label_ids = label_ids.to(device)
        input_mask = input_mask.to(device)
        # y_pred = model(input_ids, input_mask)  # 不传入labels真实标签就是预测  传入就是算损失
        loss = model(input_ids, input_mask, label_ids)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        # gradient clipping 反向传播之后进行梯度裁剪
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=clip_grad)
        optimizer.step()
        scheduler.step()
    train_loss = float(train_loss) / len(train_loader)
    logging.info("epoch: {}, train loss: {}".format(epoch, train_loss))


def evaluate(dev_loader, model, id2label, device, epoch):
    logging.info(f"evaluating...")
    # set model to eval mode
    model.eval()
    # step number in one epoch: 336
    dev_loss = 0.0
    # 存真实标签序列
    y_true_list = []
    # 存预测标签序列
    y_pred_list = []
    # 指定不进行梯度计算（没有反向传播也会计算梯度，增大GPU开销
    with torch.no_grad():
        for idx, batch_samples in enumerate(tqdm(dev_loader)):
            input_ids, label_ids, input_mask = batch_samples
            input_ids = input_ids.to(device)
            label_ids = label_ids.to(device)
            input_mask = input_mask.to(device)

            y_pred = model(input_ids, input_mask)
            loss = model(input_ids, input_mask, label_ids)
            dev_loss += loss.item()
            for lst in y_pred:
                y_pred_list.append([id2label[i] for i in lst])
            for y, m in zip(label_ids, input_mask):
                y_true_list.append([id2label[i] for i in y[m == True].tolist()])

    dev_loss = float(dev_loss) / len(dev_loader)
    logging.info("epoch: {}, dev loss: {}".format(epoch, dev_loss))
    return y_true_list, y_pred_list

def train(train_loader, dev_loader, model, id2label, optimizer, device, scheduler):
# def train(train_loader, dev_loader, model, id2label, optimizer, device):
    """train the model and evaluate model performance"""
    best_val_f1 = 0.0
    best_epoch = 0  # Initialize a variable to store the epoch of the best F1 score
    patience_counter = 0
    # start training
    for epoch in range(1, EPOCH_NUMS + 1):
        epoch_train(train_loader, model, optimizer, device, epoch, scheduler)
        # epoch_train(train_loader, model, optimizer, device, epoch)
        with torch.no_grad():
            y_true_list, y_pred_list = evaluate(dev_loader, model,id2label, device, epoch)
            report_dict = report(y_true_list, y_pred_list, output_dict=True)
            # print(report(y_true_list, y_pred_list))
            # logging.info(report(y_true_list, y_pred_list))
            # 创建表格对象
            table = PrettyTable()
            table.field_names = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
            # 填充表格数据
            for class_name, metrics in report_dict.items():
                precision = "{:.3f}".format(metrics['precision'])
                recall = "{:.3f}".format(metrics['recall'])
                f1_score = "{:.3f}".format(metrics['f1-score'])
                support = metrics['support']
                table.add_row([class_name, precision, recall, f1_score, support])

            # 输出表格
            # logging.info(table)
            for row in str(table).split('\n'):
                logging.info(row)
            val_f1 = report_dict["macro avg"]["f1-score"]
            improve_f1 = val_f1 - best_val_f1
            # 有提升
            if improve_f1 > 1e-5:
                # 如果有提升，则更新最高f1
                best_val_f1 = val_f1
                best_epoch = epoch # Update the best epoch when a new best F1 score is found
                # 这里是缓存的整个模型而不是模型的参数。
                # 检查目录是否存在，如果不存在则创建
                if not os.path.exists(MODEL_DIR):
                    os.makedirs(MODEL_DIR)
                torch.save(model.state_dict(), MODEL_DIR + f'model_{epoch}.pth')
                logging.info("Current val f1: {}".format(val_f1))
                logging.info("Best val f1: {}".format(best_val_f1))
                logging.info(f"Best val f1 was observed at epoch: {best_epoch}")
                logging.info("-------------Save best model!-------------")
                if improve_f1 < PATIENCE:
                    patience_counter += 1
                else:
                    patience_counter = 0
            # 没有提升
            else:
                logging.info("Current val f1: {}".format(val_f1))
                logging.info("Best val f1: {}".format(best_val_f1))
                logging.info(f"Best val f1 was observed at epoch: {best_epoch}")
                patience_counter += 1

            # Early stopping and logging best f1
            if (patience_counter >= PATIENCE_NUM and epoch > MIN_EPOCH_NUM) or epoch == EPOCH_NUMS:
                break
    logging.info("Training Finished!")
    logging.info("Best val f1: {}".format(best_val_f1))
    logging.info(f"Best val f1 was observed at epoch: {best_epoch}")

if __name__ == '__main__':
    set_logger(LOG_PATH)
    # 加载tokenizer 这里用的是本地模型 不能用AutoTokenizer
    tokenizer = BertTokenizer.from_pretrained(
        bert_model)  # AutoTokenizer 是 transformers 库中的一个类，用于根据指定的预训练模型自动选择并加载正确的tokenizer类和预训练权重。这个类是非常有用的，可以根据你提供的模型名称（如 "bert-base-uncased-uncased"、"gpt2" 等）自动识别并实例化相应的tokenizer。
    # 生成标签表
    vocab = Vocabulary()
    data = vocab.load_data(TRAIN_SAMPLE_PATH)
    vocab.generate_label(data)
    # 创建数据集
    train_dataset = NERDataset(mode='train', tokenizer=tokenizer, vocab=vocab, base_len=50)
    dev_dataset = NERDataset(mode='dev', tokenizer=tokenizer, vocab=vocab, base_len=50)
    train_size = len(train_dataset)
    dev_size = len(dev_dataset)
    logging.info(f"length of train_dataset:{train_size}")
    logging.info(f"length of dev_dataset:{dev_size}")

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    model = BertBiLSTMCRF(bert_model=bert_model,lstm_dim=HIDDEN_SIZE,
                    num_labels=vocab.label_size())
    model.to(DEVICE)
    logging.info(model)

    # optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    optimizer = AdamW(model.parameters(), lr=LR)
    train_steps_per_epoch = train_size // BATCH_SIZE  # 计算每个epoch的训练步数 即iterations
    # get_cosine_schedule_with_warmup 是一个创建余弦退火调度器的函数，这种调度器会在初期的预热阶段逐渐增加学习率，然后根据余弦函数逐渐减少学习率。
    # -num_warmup_steps: 预热阶段的步数。在这里，它被设置为总训练轮数（config.epoch_num）的十分之一，乘以每个epoch的训练步数。即在初始10%的训练期间，学习率会逐渐增加。
    # -num_training_steps: 总的训练步数，等于总训练轮数乘以每个epoch的训练步数。
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=(EPOCH_NUMS // 20) * train_steps_per_epoch,
                                                num_training_steps=EPOCH_NUMS * train_steps_per_epoch)
    train(train_loader=train_loader, dev_loader=dev_loader, id2label=vocab.id2label, model=model, optimizer=optimizer,
          device=DEVICE, scheduler=scheduler)
    # train(train_loader=train_loader, dev_loader=dev_loader, id2label=vocab.id2label, model=model, optimizer=optimizer,
    #       device=DEVICE)

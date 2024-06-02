from tqdm import tqdm
from utils import *
from data_loader import *
from vocabulary import Vocabulary
from config import *


# 测试集上测试效果
def test(id2label, device, test_dataset):
    """test model performance on the final test set"""

    logging.info("***** Running test *****")
    logging.info(f" Num of Test examples = {len(test_dataset)}")
    logging.info(f" Batch size = {BATCH_SIZE}")
    # build data_loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
    )
    # Prepare model
    if MODEL_DIR is not None:
        # 获取ckpt文件夹中所有文件的列表
        pth_files = os.listdir(MODEL_DIR)
        # 按照文件修改时间进行排序
        sorted_files = sorted(pth_files, key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x)), reverse=True)

        # 获取最后一个文件名
        model_path = os.path.join(MODEL_DIR, sorted_files[0])
        # model
        model = torch.load(model_path)
        model.to(device)
        logging.info("--------Load model from {}--------".format(model_path))
    else:
        logging.info("--------No model to test !--------")
        return

    model.eval()
    test_loss = 0.0
    # 存真实标签序列
    y_true_list = []
    # 存预测标签序列
    y_pred_list = []

    # 指定不进行梯度计算（没有反向传播也会计算梯度，增大GPU开销
    with torch.no_grad():
        for idx, batch_samples in enumerate(tqdm(test_loader)):
            input_ids, label_ids, input_mask = batch_samples
            input_ids = input_ids.to(device)
            label_ids = label_ids.to(device)
            input_mask = input_mask.to(device)

            y_pred = model(input_ids, input_mask)
            loss = model(input_ids, input_mask, label_ids)
            test_loss += loss.item()
            for lst in y_pred:
                y_pred_list.append([id2label[i] for i in lst])
            for y, m in zip(label_ids, input_mask):
                y_true_list.append([id2label[i] for i in y[m == True].tolist()])

    test_loss = float(test_loss) / len(test_loader)
    logging.info("test loss: {}".format(test_loss))
    logging.info(report(y_true_list, y_pred_list, output_dict=False))


if __name__ == '__main__':
    set_logger(LOG_PATH)
    vocab = Vocabulary()
    id2label, _ = vocab.get_label()  # eval时会用到id2label
    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        bert_model)
    test_dataset = NERDataset(mode='test',vocab=vocab, tokenizer=tokenizer)
    test(id2label=id2label, device=DEVICE, test_dataset=test_dataset)

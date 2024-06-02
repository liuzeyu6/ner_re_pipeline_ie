import logging
import os
from config import *
from seqeval.metrics import classification_report

'''
评估函数  计算precision recall f1并打印
'''
def report(y_true, y_pred, output_dict=False):
    # return classification_report(y_true, y_pred)  这种是直接输出str类型的报告  下面是输出字典
    return classification_report(y_true, y_pred, output_dict=output_dict)

def set_logger(log_path=LOG_PATH):
    """Set the logger to logger info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    创建一个日志记录器并配置两个处理器，分别用于将日志写入文件和在控制台上打印
    in a permanent file. Here we save it to `model_dir/train.logger`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to logger
    """
    if os.path.exists(log_path):
        os.remove(log_path)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    #检查logger是否有已存在的处理器（handler）
    if not logger.handlers:
        # 1 Logging to a file 创建一个文件处理器（file_handler），用于将日志写入指定路径的文件。
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s')) #按照时间、日志级别和消息的格式进行记录。
        logger.addHandler(file_handler) #将文件处理器添加到日志记录器中，以便将日志写入文件

        # 2 Logging to console 创建一个流处理器（stream_handler），用于将日志输出到控制台。
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s')) #这种格式化字符串将在流处理器中使用，以便将日志记录输出到控制台上。由于控制台不需要显示时间和日志级别等其他信息，因此只保留了日志消息的内容。
        logger.addHandler(stream_handler)


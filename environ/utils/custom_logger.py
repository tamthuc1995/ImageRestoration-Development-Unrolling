import logging


all_avalable_logger = {}

def get_root_logger(logger_name='experiment_example', log_level=logging.INFO, log_file=None):

    logger = logging.getLogger(logger_name)
    # if the logger has been initialized, just return it
    if logger_name in all_avalable_logger:
        return logger

    format_str = '%(asctime)s %(levelname)s: %(message)s'
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(stream_handler)
    logger.setLevel(log_level)
    # add file handler
    file_handler = logging.FileHandler(log_file, 'w')
    file_handler.setFormatter(logging.Formatter(format_str))
    file_handler.setLevel(log_level)
    logger.addHandler(file_handler)
    all_avalable_logger[logger_name] = True

    return logger


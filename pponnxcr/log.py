import logging

logger = logging.getLogger('pponnxcr')
logger.addHandler(logging.StreamHandler())

def get_logger(name):
    return logging.getLogger('pponnxcr.'+name)

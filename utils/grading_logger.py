'''
My logger. For good logging to file or to console
'''

import logging  
import time
import os
import sys 

def get_logger(file_mode=True, logger_name="detail", dir_name='runs/Logs13', detailedConsoleHandler=False):
    logger = logging.getLogger(logger_name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)  
    # logger.handlers = []

    # set formatter
    rq = time.strftime('%m%d-%H:%M:%S', time.localtime(time.time()))
    formatter = logging.Formatter("%(asctime)s - %(filename)s [line:%(lineno)d] - %(levelname)s: %(message)s")

    if file_mode == True:
        # I. add a file handler  
        if not os.path.exists(dir_name) or os.path.isfile(dir_name):
            os.makedirs(dir_name)
        detail_log_name = os.path.join(dir_name, rq) + '.log'
        fh = logging.FileHandler(detail_log_name, mode='w')
        fh.setLevel(logging.INFO)  
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # II. add a console handler 
    ch = logging.StreamHandler(stream=sys.stdout)
    chlevel = logging.DEBUG if detailedConsoleHandler else logging.WARNING  
    ch.setLevel(chlevel) 
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
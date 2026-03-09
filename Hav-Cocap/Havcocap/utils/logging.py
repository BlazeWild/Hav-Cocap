# -*- coding: utf-8 -*-
# @Time    : 2022/11/12 22:32
# @Author  : Yaojie Shen
# @Project : CoCap
# @File    : logging.py


import logging
import os

import colorlog
import torch.distributed as dist

level_dict = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
    "notset": logging.NOTSET
}


# noinspection SpellCheckingInspection
def setup_logging(cfg=None, output_dir=None):
    logger = logging.getLogger('Havcocap') # Use a common root logger
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%m/%d %H:%M:%S")
    
    # Console handler
    handler_console = logging.StreamHandler()
    handler_console.setLevel(logging.INFO)
    handler_console.setFormatter(formatter)
    logger.addHandler(handler_console)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, "train.log")
        handler_file = logging.FileHandler(log_file, mode="a")
        handler_file.setLevel(logging.INFO)
        handler_file.setFormatter(formatter)
        logger.addHandler(handler_file)
        return logger

    if cfg:
        # log file
        if len(str(cfg.LOG.LOGGER_FILE).split(".")) == 2:
            file_name, extension = str(cfg.LOG.LOGGER_FILE).split(".")
            log_file_debug = os.path.join(cfg.LOG.DIR, f"{file_name}_debug.{extension}")
            log_file_info = os.path.join(cfg.LOG.DIR, f"{file_name}_info.{extension}")
        elif len(str(cfg.LOG.LOGGER_FILE).split(".")) == 1:
            file_name = cfg.LOG.LOGGER_FILE
            log_file_debug = os.path.join(cfg.LOG.DIR, f"{file_name}_debug")
            log_file_info = os.path.join(cfg.LOG.DIR, f"{file_name}_info")
        else:
            raise ValueError("cfg.LOG.LOGGER_FILE is invalid: %s", cfg.LOG.LOGGER_FILE)
            
        # log file
        if os.path.dirname(log_file_debug):  # dir name is not empty
            os.makedirs(os.path.dirname(log_file_debug), exist_ok=True)
            
        # console
        # Assuming cfg has these attributes, otherwise wrap in try/except or remove if not needed for this port
        try:
            handler_console.setLevel(level_dict[cfg.LOG.LOGGER_CONSOLE_LEVEL.lower()])
        except:
            pass

        # debug level
        handler_debug = logging.FileHandler(log_file_debug, mode="a")
        handler_debug.setLevel(logging.DEBUG)
        handler_debug.setFormatter(formatter)
        logger.addHandler(handler_debug)
        # info level
        handler_info = logging.FileHandler(log_file_info, mode="a")
        handler_info.setLevel(logging.INFO)
        handler_info.setFormatter(formatter)
        logger.addHandler(handler_info)

    logger.propagate = False
    return logger


def show_registry():
    from cocap.data.build import DATASET_REGISTRY, COLLATE_FN_REGISTER
    from cocap.modeling.model import MODEL_REGISTRY
    from cocap.modeling.optimizer import OPTIMIZER_REGISTRY
    from cocap.modeling.loss import LOSS_REGISTRY
    from cocap.modeling.meter import METER_REGISTRY

    logger = logging.getLogger(__name__)
    logger.debug(DATASET_REGISTRY)
    logger.debug(COLLATE_FN_REGISTER)
    logger.debug(MODEL_REGISTRY)
    logger.debug(OPTIMIZER_REGISTRY)
    logger.debug(LOSS_REGISTRY)
    logger.debug(METER_REGISTRY)

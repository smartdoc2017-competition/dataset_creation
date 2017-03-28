#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================================================
# Imports
import logging

# ==============================================================================
# logger = logging.getLogger(__name__)

# ==============================================================================
DBGLINELEN = 80
DBGSEP = "-"*DBGLINELEN


def programHeader(logger, prog_name, prog_version):
    logger.debug(DBGSEP)
    dbg_head = "%s - v. %s" % (prog_name, prog_version)
    dbg_head_pre = " " * (max(0, (DBGLINELEN - len(dbg_head)))/2)
    logger.debug(dbg_head_pre + dbg_head)

def initLogger(logger, debug=False):
    format="%(name)-12s %(levelname)-7s: %(message)s" #%(module)-10s
    formatter = logging.Formatter(format)    
    ch = logging.StreamHandler()  
    ch.setFormatter(formatter)  
    logger.addHandler(ch)
    level = logging.INFO
    if debug:
        level = logging.DEBUG
    logger.setLevel(level)


def createLogger(name):
    return logging.getLogger(name)

    
def createAndInitLogger(name, debug=False):
    logger = createLogger(name)
    initLogger(logger, debug)
    return logger


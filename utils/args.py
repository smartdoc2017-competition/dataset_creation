#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================================================
# Imports
import os.path
import argparse

# ==============================================================================
class StoreValidFilePath(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not os.path.isfile(values):
            parser.error("'%s' does not exist or is not a file." % values)
        setattr(namespace, self.dest, values)

class StoreExistingOrCreatableDir(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if type(values) is not str:
            parser.error("'%s' does not represent a directory path." % values)
        if os.path.exists(values) and not os.path.isdir(values):
            parser.error("'%s' is not a directory." % values)
        if os.path.exists(values) and not os.access(values, os.W_OK):
            parser.error("'%s' is not writable." % values)
        if not os.path.exists(values):
            try:
                os.makedirs(values)
            except OSError as e: 
                parser.error("'%s' cannot be created." % values)
        setattr(namespace, self.dest, values)

def dumpArgs(args, logger):
    logger.debug("Arguments:")
    for (k, v) in args.__dict__.items():
        logger.debug("    %-20s = %s" % (k, v))

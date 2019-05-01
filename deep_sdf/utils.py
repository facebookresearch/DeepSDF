#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import logging

def add_common_args(arg_parser):
    arg_parser.add_argument('--debug', dest='debug', default=False, action='store_true', help=
        'If set, debugging messages will be printed')
    arg_parser.add_argument('--quiet', '-q', dest='quiet', default=False, action='store_true', help=
        'If set, only warnings will be printed')

def configure_logging(args):
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.WARNING)
    logger_handler = logging.StreamHandler()
    logger_handler.setFormatter(logging.Formatter('DeepSdf - %(levelname)s - %(message)s)'))
    logger.addHandler(logger_handler)

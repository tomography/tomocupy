#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2022, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2015. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################

'''Customized logging for the tomocupy library.

Logging in tomocupy is built upon the standard logging functionality
in python. This module provides the standard ``getLogger`` function
that can be used to get a logger object with the usual *debug*,
*info*, etc., methods. If ``setup_custom_logger`` is called, all
``tomocupy.*`` loggers will use color terminal logging and/or a
logfile.

'''
import traceback
import logging
from logging import *


__all__ = ['setup_custom_logger', 'ColoredLogFormatter'] + logging.__all__


def log_exception(logger, err, fmt="%s"):
    """Send a reconstructed stacktrace to the log.

    The stacktrace will be sent to the error log for the given logger.

    Parameters
    ==========
    logger
      A logger, as returned by ``logging.getLogger(...)``
    err
      An exception to log.
    fmt
      Logging format string for each line of the exception
      (e.g. "  *** %s"")
    
    """
    tb_lines = traceback.format_exception(type(err), err, err.__traceback__)
    tb_lines = [ln for lns in tb_lines for ln in lns.splitlines()]
    for tb_line in tb_lines:
        logger.error("      %s", tb_line)


def setup_custom_logger(lfname: str=None, stream_to_console: bool=True, level=logging.DEBUG):
    """Prepare the logging system with custom formatting.
    
    This adds handlers to the *tomocupy* parent logger. Any logger
    inside tomocupy will produce output based on this functions
    customization parameters. The file given in *lfname* will receive
    all log message levels, while the console will receive messages
    based on *level*.
    
    Parameters
    ----------
    lfname
      Path to where the log file should be stored. If omitted, no file
      logging will be performed.
    stream_to_console
      If true, logs will be output to the console with color
      formatting.
    level
      A logging level for the stream handler. This can be either a
      string ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"), or an
      actual level defined in the python logging framework.

    """
    parent_name = __name__.split('.')[0]  # Nominally "tomocupy"
    parent_logger = logging.getLogger(parent_name)
    parent_logger.setLevel(logging.DEBUG)
    # Set up normal output to a file
    if lfname is not None:
        fHandler = logging.FileHandler(lfname)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s(%(lineno)s) - %(levelname)s: %(message)s')
        fHandler.setFormatter(file_formatter)
        fHandler.setLevel(logging.DEBUG)
        parent_logger.addHandler(fHandler)
    # Set up formatted output to the console
    if stream_to_console:
        ch = logging.StreamHandler()
        ch.setFormatter(ColoredLogFormatter('%(asctime)s - %(message)s'))
        ch.setLevel(level)
        parent_logger.addHandler(ch)


class ColoredLogFormatter(logging.Formatter):
    """A logging formatter that add console color codes."""
    __BLUE = '\033[94m'
    __GREEN = '\033[92m'
    __RED = '\033[91m'
    __RED_BG = '\033[41m'
    __YELLOW = '\033[33m'
    __ENDC = '\033[0m'
    
    def _format_message_level(self, message, level):
        colors = {
            'INFO': self.__GREEN,
            'WARNING': self.__YELLOW,
            'ERROR': self.__RED,
            'CRITICAL': self.__RED_BG,
        }
        if level in colors.keys():
            message = "{color}{message}{ending}".format(color=colors[level],
                                                        message=message,
                                                        ending=self.__ENDC)
        return message
    
    def formatMessage(self, record):
        record.message = self._format_message_level(record.message, record.levelname)
        return super().formatMessage(record)
import os
import sys
import shutil
from copy import copy
from pathlib import Path
import argparse
import configparser
from collections import OrderedDict
import contextlib
import logging
import warnings
import inspect

import h5py
import numpy as np

from h5gpurec import utils


log = logging.getLogger(__name__)


def default_parameter(func, param):
    """Get the default value for a function parameter.
    
    For a given function *func*, introspect the function and return
    the default value for the function parameter named *param*.
    
    Return
    ======
    default_val
      The default value for the parameter.
    
    Raises
    ======
    RuntimeError
      Raised if the function *func* has no default value for the
      requested parameter *param*.
    
    """
    # Retrieve the function parameter by introspection
    try:
        sig = inspect.signature(func)
        _param = sig.parameters[param]
    except TypeError as e:
        warnings.warn(str(e))
        log.warning(str(e))
        return None
    # Check if a default value exists
    if _param.default is _param.empty:
        # No default is listed in the function, so throw an exception
        msg = ("No default value given for parameter *{}* of callable {}."
               "".format(param, func))
        raise RuntimeError(msg)
    else:
        # Retrieve and return the parameter's default value
        return _param.default


LOGS_HOME = Path.home()/'logs'
CONFIG_FILE_NAME = Path.home()/'h5gpurecon.conf'

SECTIONS = OrderedDict()


SECTIONS['general'] = {
    'config': {
        'default': CONFIG_FILE_NAME,
        'type': str,
        'help': "File name of configuration file",
        'metavar': 'FILE'},
    'logs-home': {
        'default': LOGS_HOME,
        'type': str,
        'help': "Log file directory",
        'metavar': 'FILE'},
    'verbose': {
        'default': False,
        'help': 'Verbose output',
        'action': 'store_true'},
    'config-update': {
        'default': False,
        'help': 'When set, the content of the config file is updated using the current params values',
        'action': 'store_true'},
        }




SECTIONS['file-reading'] = {
    'file-name': {
        'default': '.',
        'type': Path,
        'help': "Name of the last used hdf file or directory containing multiple hdf files",
        'metavar': 'PATH'},
    'out-path-name': {
        'default': None,
        'type': Path,
        'help': "Name of the last used hdf file or directory containing multiple hdf files",
        'metavar': 'PATH'},
    'file-type': {
        'default': 'standard',
        'type': str,
        'help': "Input file type",
        'choices': ['standard', 'double_fov']},        
    'nsino-per-chunk': {     
        'type': int,
        'default': 8,
        'help': "Number of sinograms per chunk. Use larger numbers with computers with larger memory.  Value <= 0 defaults to # of cpus.",},
    'binning': {
        'type': utils.positive_int,
        'default': 0,
        'help': "Reconstruction binning factor as power(2, choice)",
        'choices': [0, 1, 2, 3]},
       }

SECTIONS['remove-stripe'] = {
    'remove-stripe-method': {
        'default': 'none',
        'type': str,
        'help': "Remove stripe method: none, fourier-wavelet",
        'choices': ['none', 'fw']},
        }

# pixel_size=(params.pixel_size*1e-4),dist=(params.propagation_distance/10.0),energy=params.energy, alpha=params.retrieve_phase_alpha,pad=True
SECTIONS['retrieve-phase'] = {
    'retrieve-phase-method': {
        'default': 'none',
        'type': str,
        'help': "Retrieve phase method: none, paganin. 'dist' and 'pixel_size' in centimeters, 'energy' in keV, 'alpha' for regularization.",
        'choices': ['none', 'paganin']},
    'dist' :{
        'default' : 100.0,
        'type' : float,
        'help' : ''},
    'pixel-size' :{
        'default' : 1.0e-4,
        'type' : float,
        'help' : ''},
    'energy' :{
        'default' : 25.5,
        'type' : float,
        'help' : ''},
    'alpha' :{
        'default' : 1.0e-3,
        'type' : float,
        'help' : ''},
    'pad': {
        'default': True,
        'help': '',
        'action': 'store_true'}, #CHECK - what is store_true? is there store_false?#
        }


SECTIONS['reconstruction'] = {
    'reconstruction-type': {
        'default': 'try',
        'type': str,
        'help': "Reconstruct full data set. ",
        'choices': ['full','try']},
    'reconstruction-algorithm': {
        'default': 'fourierrec',
        'type': str,
        'help': "Reconstruction algorithm",
        'choices': ['fourierrec', 'lprec']},    
    'rotation-axis': {
        'default': -1.0,
        'type': float,
        'help': "Location of rotation axis"},
    'center-search-width': {
        'type': float,
        'default': 10.0,
        'help': "+/- center search width (pixel). "},
    'center-search-step': {
        'type': float,
        'default': 0.5,
        'help': "+/- center search step (pixel). "},
    'nsino': {
        'default': 0.5,
        'type': float,
        'help': 'Location of the sinogram used for slice reconstruction and find axis (0 top, 1 bottom)'},
    'blocked-views': {
        'default': False,
        'help': 'When set, the blocked-views options are used',
        'action': 'store_true'},    
    'start-row': {
        'type': int,
        'default': 0,
        'help': "Start slice"},
    'end-row': {
        'type': int,
        'default': -1,
        'help': "End slice"},
    'start-proj': {
        'type': int,
        'default': 0,
        'help': "Start projection"},
    'end-proj': {
        'type': int,
        'default': -1,
        'help': "End projection"},
    }

SECTIONS['blocked-views'] = {
    'blocked-views-start': {
        'type': float,
        'default': 0,
        'help': "Angle of the first blocked view"},
    'blocked-views-end': {
        'type': float,
        'default': 1,
        'help': "Angle of the last blocked view"},
        }

RECON_PARAMS = ('file-reading', 'remove-stripe',  'reconstruction', 'blocked-views', 'retrieve-phase')

NICE_NAMES = ('General', 'File reading', 'Remove stripe', 'Reconstruction')

def get_config_name():
    """Get the command line --config option."""
    name = CONFIG_FILE_NAME
    for i, arg in enumerate(sys.argv):
        if arg.startswith('--config'):
            if arg == '--config':
                return sys.argv[i + 1]
            else:
                name = sys.argv[i].split('--config')[1]
                if name[0] == '=':
                    name = name[1:]
                return name

    return name


def parse_known_args(parser, subparser=False):
    """
    Parse arguments from file and then override by the ones specified on the
    command line. Use *parser* for parsing and is *subparser* is True take into
    account that there is a value on the command line specifying the subparser.
    """
    if len(sys.argv) > 1:
        subparser_value = [sys.argv[1]] if subparser else []
        config_values = config_to_list(config_name=get_config_name())
        values = subparser_value + config_values + sys.argv[1:]
    else:
        values = ""

    return parser.parse_known_args(values)[0]


def config_to_list(config_name=CONFIG_FILE_NAME):
    """
    Read arguments from config file and convert them to a list of keys and
    values as sys.argv does when they are specified on the command line.
    *config_name* is the file name of the config file.
    """
    result = []
    config = configparser.ConfigParser()

    if not config.read([config_name]):
        return []

    for section in SECTIONS:
        for name, opts in ((n, o) for n, o in SECTIONS[section].items() if config.has_option(section, n)):
            value = config.get(section, name)

            if value != '' and value != 'None':
                action = opts.get('action', None)

                if action == 'store_true' and value == 'True':
                    # Only the key is on the command line for this action
                    result.append('--{}'.format(name))

                if not action == 'store_true':
                    if opts.get('nargs', None) == '+':
                        result.append('--{}'.format(name))
                        result.extend((v.strip() for v in value.split(',')))
                    else:
                        result.append('--{}={}'.format(name, value))

    return result
   

class Params(object):
    def __init__(self, sections=()):
        self.sections = sections + ('general', )

    def add_parser_args(self, parser):
        for section in self.sections:
            for name in sorted(SECTIONS[section]):
                opts = SECTIONS[section][name]
                parser.add_argument('--{}'.format(name), **opts)

    def add_arguments(self, parser):
        self.add_parser_args(parser)
        return parser

    def get_defaults(self):
        parser = argparse.ArgumentParser()
        self.add_arguments(parser)

        return parser.parse_args('')

def write(config_file, args=None, sections=None):
    """
    Write *config_file* with values from *args* if they are specified,
    otherwise use the defaults. If *sections* are specified, write values from
    *args* only to those sections, use the defaults on the remaining ones.
    """
    config = configparser.ConfigParser()

    for section in SECTIONS:
        config.add_section(section)
        for name, opts in SECTIONS[section].items():
            if args and sections and section in sections and hasattr(args, name.replace('-', '_')):
                value = getattr(args, name.replace('-', '_'))

                if isinstance(value, list):
                    value = ', '.join(value)
            else:
                value = opts['default'] if opts['default'] is not None else ''

            prefix = '# ' if value == '' else ''

            if name != 'config':
                config.set(section, prefix + name, str(value))

    with open(config_file, 'w') as f:
        config.write(f)

def show_config(args):
    """Log all values set in the args namespace.
    Arguments are grouped according to their section and logged alphabetically
    using the DEBUG log level thus --verbose is required.
    """
    args = args.__dict__

    log.warning('h5gpurec status start')
    for section, name in zip(SECTIONS, NICE_NAMES):
        entries = sorted((k for k in args.keys() if k.replace('_', '-') in SECTIONS[section]))
        if entries:
            for entry in entries:
                value = args[entry] if args[entry] != None else "-"
                log.info("  {:<16} {}".format(entry, value))

    log.warning('h5gpurec status end')

def log_values(args):
    """Log all values set in the args namespace.
    Arguments are grouped according to their section and logged alphabetically
    using the DEBUG log level thus --verbose is required.
    """
    args = args.__dict__

    log.warning('h5gpurecon status start')
    for section, name in zip(SECTIONS, NICE_NAMES):
        entries = sorted((k for k in args.keys() if k.replace('_', '-') in SECTIONS[section]))

        # print('log_values', section, name, entries)
        if entries:
            log.info(name)

            for entry in entries:
                value = args[entry] if args[entry] is not None else "-"
                if (value == 'none'):
                    log.warning("  {:<16} {}".format(entry, value))
                elif (value is not False):
                    log.info("  {:<16} {}".format(entry, value))
                elif (value is False):
                    log.warning("  {:<16} {}".format(entry, value))

    log.warning('h5gpurecon status end')


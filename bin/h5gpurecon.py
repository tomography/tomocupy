import sys
import h5py
import numpy as np
import time
import argparse
import os
from pathlib import Path
from datetime import datetime
from h5gpurec import H5GPURec
from h5gpurec import logging
from h5gpurec import config

log = logging.getLogger('h5gpurec.bin.h5gpurecon')


def init(args):
    if not os.path.exists(str(args.config)):
        config.write(args.config)
    else:
        log.error("{0} already exists".format(args.config))    

def run_status(args):
    config.log_values(args)

def run_rec(args):
    config.show_config(args)
    file_name = Path(args.file_name)    
    if file_name.is_file():        
        t = time.time()
        clpthandle = H5GPURec(args)        
        if(args.reconstruction_type=='full'):
            clpthandle.recon_all()
        if(args.reconstruction_type=='try'):
            clpthandle.recon_all_try()
        print(f'Reconstruction time {(time.time()-t):.01f}s')
    else:
        log.error("File Name does not exist: %s" % args.file_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', **config.SECTIONS['general']['config'])    
    tomo_params = config.RECON_PARAMS
    
    cmd_parsers = [
        ('init',        init,            (),                             "Create configuration file"),
        ('recon',       run_rec,         tomo_params,                    "Run tomographic reconstruction"),
        ('status',      run_status,      tomo_params,                    "Show the tomographic reconstruction status"),        
    ]

    subparsers = parser.add_subparsers(title="Commands", metavar='')

    for cmd, func, sections, text in cmd_parsers:
        cmd_params = config.Params(sections=sections)
        cmd_parser = subparsers.add_parser(cmd, help=text, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        cmd_parser = cmd_params.add_arguments(cmd_parser)
        cmd_parser.set_defaults(_func=func)

    args = config.parse_known_args(parser, subparser=True)
    # create logger
    logs_home = args.logs_home

    # make sure logs directory exists
    if not os.path.exists(logs_home):
        os.makedirs(logs_home)

    lfname = os.path.join(logs_home, 'h5gpurecon_' + datetime.strftime(datetime.now(), "%Y-%m-%d_%H_%M_%S") + '.log')
    log_level = 'DEBUG' if args.verbose else "INFO"
    logging.setup_custom_logger(lfname, level=log_level)
    log.debug("Started h5gpurecon")
    log.info("Saving log at %s" % lfname)

    try:
        args._func(args)
    except RuntimeError as e:
        log.error(str(e))
        sys.exit(1)


if __name__ == '__main__':
    main()




if __name__ == '__main__':
    main()

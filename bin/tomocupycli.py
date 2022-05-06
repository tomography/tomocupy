import sys
import h5py
import numpy as np
import time
import argparse
import os
import subprocess
from pathlib import Path
from datetime import datetime

from tomocupy_cli import logging
from tomocupy_cli import config
from tomocupy_cli import GPURec
from tomocupy_cli import GPURecSteps

log = logging.getLogger(__name__)


def init(args):
    if not os.path.exists(str(args.config)):
        config.write(args.config)
    else:
        log.error("{0} already exists".format(args.config))    

def run_status(args):
    config.log_values(args)

def run_rec(args):
    #config.show_config(args)
    file_name = Path(args.file_name)    
    if file_name.is_file():        
        t = time.time()
        clpthandle = GPURec(args)        
        if(args.reconstruction_type=='full'):
            clpthandle.recon_all()
        if(args.reconstruction_type=='try'):
            clpthandle.recon_all_try()
        log.warning(f'Reconstruction time {(time.time()-t):.01f}s')
    else:
        log.error("File Name does not exist: %s" % args.file_name)

def run_recstep(args):
    #config.show_config(args)
    file_name = Path(args.file_name)    
    if file_name.is_file():        
        t = time.time()
        clpthandle = GPURecSteps(args)        
        clpthandle.recon_steps()
        log.warning(f'Reconstruction time {(time.time()-t):.01f}s')
    else:
        log.error("File Name does not exist: %s" % args.file_name)

def run_recmulti(args):
    line = ' '.join(sys.argv[2:])
    if(args.end_row==-1):
        with h5py.File(args.file_name,'r') as fid:
           args.end_row = fid['/exchange/data/'].shape[1]
    
    cmd1 = f"ssh -t tomo@tomo1 \"bash -c 'source ~/.bashrc; conda activate tomocupyfp16; tomocupyfp16 recon {line} --start-row {args.start_row} --end-row {args.end_row//2}\';\""
    cmd2 = f"ssh -t tomo@tomo2 \"bash -c 'source ~/.bashrc; conda activate tomocupyfp16; tomocupyfp16 recon {line} --start-row {args.end_row//2} --end-row {args.end_row}\'; \""
    print(f'Tomo1: {cmd1}')
    p1 = subprocess.Popen(cmd1,shell=True)
    print(f'Tomo2: {cmd2}')
    p2 = subprocess.Popen(cmd2,shell=True)
    p1.wait()
    p2.wait()  
    # recover terminal
    os.system('stty sane')
    
def defaults_from_dxchange(file_name):

    hf = h5py.File(file_name)
    energy_mode = int(hf['measurement/instrument/monochromator/energy_mode'][:])
    if energy_mode < 2:
        energy = float(hf['measurement/instrument/monochromator/energy'][:])
    else:
        energy = 50.0 # what should be white-beam mean energy?

    pixel_size = float(hf['measurement/instrument/detection_system/objective/resolution'][:])
    dist = float(hf['measurement/instrument/camera_motor_stack/setup/camera_distance'][:])
    hf.close()
    meta_dict = {"pixel_size" : pixel_size, "propagation_distance" : dist, "energy" : energy}
    print(f'DEBUG: {meta_dict}')
    return meta_dict

def get_file_name():
    """Get the command line --config option."""
    for i, arg in enumerate(sys.argv):
        if arg.startswith('--file-name'):
            if arg == '--file-name':
                return sys.argv[i + 1]
            else:
                name = sys.argv[i].split('--file-name')[1]
                if name[0] == '=':
                    name = name[1:]
                return name
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', **config.SECTIONS['general']['config'])    
    tomo_params = config.RECON_PARAMS
    tomo_steps_params = config.RECON_STEPS_PARAMS
    #
    
    cmd_parsers = [
        ('init',        init,            (),                             "Create configuration file"),
        ('recon',       run_rec,         tomo_params,                    "Run tomographic reconstruction by splitting data into chunks in z "),
        ('reconstep',   run_recstep,     tomo_steps_params,                    "Run tomographic reconstruction by splitting by chunks in z and angles (step-wise)"),
        ('reconmulti',  run_recmulti,    tomo_params,                                   "Run reconstruction on several nodes"),        
        ('status',      run_status,      tomo_steps_params,                    "Show the tomographic reconstruction status"),        
    ]

    file_name = get_file_name()
    dxchange_defaults = defaults_from_dxchange(file_name)
    subparsers = parser.add_subparsers(title="Commands", metavar='')

    for cmd, func, sections, text in cmd_parsers:
        cmd_params = config.Params(sections=sections)
        cmd_parser = subparsers.add_parser(cmd, help=text, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        cmd_parser = cmd_params.add_arguments(cmd_parser)
        cmd_parser.set_defaults(_func=func)
        cmd_parser.set_defaults(**dxchange_defaults)

    args = config.parse_known_args(parser, subparser=True) # need to do this twice so as to get file-name in advance??
    print(f'DEBUG: energy {args.energy}, pixel_size {args.pixel_size}')

    
    # create logger
    logs_home = args.logs_home

    # make sure logs directory exists
    if not os.path.exists(logs_home):
        os.makedirs(logs_home)

    lfname = os.path.join(logs_home, 'tomocupyfp16on_' + datetime.strftime(datetime.now(), "%Y-%m-%d_%H_%M_%S") + '.log')
    log_level = 'DEBUG' if args.verbose else "INFO"
    logging.setup_custom_logger(lfname, level=log_level)
    log.debug("Started tomocupyfp16on")
    log.info("Saving log at %s" % lfname)

    try:
        args._func(args)
    except RuntimeError as e:
        log.error(str(e))
        sys.exit(1)




if __name__ == '__main__':
    main()
